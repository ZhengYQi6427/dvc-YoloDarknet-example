import os
import json
import numpy as np
import pandas as pd
import multiprocessing

from copy import deepcopy
from scipy.optimize import linear_sum_assignment


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Multiprocessing workaround
    https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
    :param f: function
    :param X: list of input parameters over which to loop function
    :param nprocs: number of processors
    :return: list of outputs for each function
    """
    def fun(f, q_in, q_out):
        """
        Helper function for multiprocessing workaround
        https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
        """
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


class DetectionMeasurement:
    """
    This class can be used to measure object detection algorithm performance using metrics
     standardized by the VOC Challenge: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
     and refined by the COCO Consortium: http://cocodataset.org/#detection-eval

    Typical usage:
    - instantiate class
    - call get_all_metrics() to return mAP and AP50
    - ex:
        d = DetectionMeasurement(actual_data, predicted_data, 'maskrcnn')  # instantiate using default values for most params
        metrics = d.get_all_metrics()
        metrics['mAP']  # check out mean average precision
        metrics['mIoU']['car']  # check out mean IoU value for all matches in the 'car' class

    """

    def __init__(self, actuals, predictions, prediction_format, actuals_format='cvat', frame_numbers=None,
                 primary_iou_threshold=0.5, iou_thresholds=None, recall_levels=np.linspace(0.0, 1.0, 11)):
        """

        :param actuals: actual labels and bounding box measurements in
                        dict/json of computer vision annotation tool (cvat)
                        standard output format. See https://github.com/opencv/cvat
                        Note that object_ids must be numeric.
        :param predictions: predicted labels and bounding box measurements in the
                            standard output format of whatever model created them
                            For Supervisely outputs, this will be a path to a
                            directory with internal files named by frame number
        :param prediction_format: string identifying the format of the prediction data. Must be one of:
                                  ['maskrcnn', 'supervisely', 'zklab_measurement']. List of options can only be updated
                                  if method _build_predictions() is updated to properly convert the data.
        :param actuals_format: string identifying the format of the actuals data. Must be one of:
                               ['cvat', 'zklab_measurement']. List of options can only be updated if method
                               _build_actuals() is updated to properly convert the data.
        :param frame_numbers: list of float, identifies the frames to look at in both the actuals and predictions
        :param primary_iou_threshold: float value denoting lower bound for a valid Jaccard index (i.e.
                                      intersection over union) for an actual object and a predicted bounding box.
                                      Comparisons of predictions to actuals that yield deltas less than this
                                      value render the prediction an invalid identification of that actual.
                                      Default value is taken from VOC Challenge standard
        :param iou_thresholds: list of float values denoting all iou thresholds to use in computing mean average
                               precision (mAP), as done for COCO, where default is np.linspace(0.05, 0.95, 10)
                               when left empty, the list will consist only of the primary_iou_threshold
        :param recall_levels: list of float values denoting all recall levels to use in computing average precision
                              default comes from VOC; a second option is the COCO approach of np.linspace(0.0, 1.0, 101)
        """
        # validate IoU & recall inputs
        self.iou_thresholds = [float(x) for x in iou_thresholds] if iou_thresholds else [float(primary_iou_threshold)]
        self.recall_levels = list(recall_levels)
        values = [primary_iou_threshold] + self.iou_thresholds + self.recall_levels
        for val in values:
            if not 0 <= val <= 1:
                raise ValueError("All IoU thresholds and recall levels must be between 0 and 1.")

        # params used to determine valid matches
        self.primary_iou_threshold = float(primary_iou_threshold)
        self.class_names = ['bg', 'pedestrian', 'vehicle']
        self.recall_levels.sort()

        # numeric value to be used in the cost matrix of actual-prediction distances when their pairing is not valid
        # needs to be sufficiently high so as to eliminate the chance that this pairing is chosen in the cost minimization
        self.default_high = 100.0

        # extract relevant data about actuals
        self.actuals = {}
        self.frame_numbers = [float(x) for x in frame_numbers] if frame_numbers else []
        self.total_objects_seen = 0  # count of object observances across all frames
        self._build_actuals(actuals, actuals_format, frame_numbers)
        # extract relevant data about predictions
        self._next_object_id = -1
        self.predictions = {}
        self._build_predictions(predictions, prediction_format)

        # initialize key variables
        self.matches = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to (actual object_id, predicted object_id) to distance between actual and pred
        self.misses = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to list of object_ids
        self.false_positives = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to list of object_ids
        self.results_lists = []

        # determine matches and errors
        for iou_threshold in self.iou_thresholds:
            self._match_objects(iou_threshold)
            self.results_df = pd.DataFrame(self.results_lists, columns=['iou_threshold', 'frame_number', 
                                                                        'class_name', 'actual', 'detection', 
                                                                        'confidence', 'TP', 'FP', 'FN'])
            self._compute_errors(iou_threshold)

    def _get_next_object_id(self):
        """Provides a unique ID to be used for an object

        :return: int
        """
        self._next_object_id += 1
        return self._next_object_id

    def _build_actuals(self, actuals, actuals_format, frame_numbers):
        if actuals_format == 'cvat':
            for obj in actuals['track']:
                class_name = obj['_label'].lower()
                object_id = float(obj['_id'])
                for box in obj['box']:
                    self.total_objects_seen += 1
                    frame_number = float(box['_frame'])
                    if not frame_numbers or len(frame_numbers) == 0 or frame_number in self.frame_numbers:
                        box_data = {
                            'xtl': float(box['_xtl']),
                            'ytl': float(box['_ytl']),
                            'xbr': float(box['_xbr']),
                            'ybr': float(box['_ybr'])
                        }
                        if frame_number not in self.frame_numbers:
                            self.frame_numbers.append(frame_number)
                        if (frame_number, class_name) not in self.actuals:
                            self.actuals[(frame_number, class_name)] = {}
                        self.actuals[(frame_number, class_name)][object_id] = deepcopy(box_data)
        elif actuals_format == 'zklab_measurement':
            for (frame_index, class_name), objects in actuals.items():
                self.frame_numbers.append(frame_index)
                for _, obj in objects.items():
                    self.total_objects_seen += 1
            self.actuals = actuals
        else:
            raise NotImplementedError

    def _build_predictions(self, predictions, prediction_format):
        """Builds the dictionary of predictions in the following format:
        dict of float (frame number) to float (object_id) to
        string ('xtl', 'ytl', 'xbr', 'ybr') to float (coordinate value)
        Also ensures that object_ids are all unique from those used in actuals data

        :param predictions: predicted labels and bounding box measurements in the
                            standard output format of whatever model created them
                            For Supervisely outputs, this will be a path to a
                            directory with internal files named by frame number
        :param prediction_format: string identifying the format of the prediction data

        :return: does not return; updates self.predictions
        """
        if prediction_format == 'maskrcnn':
            for frame_number in self.frame_numbers:
                if frame_number in predictions:
                    bounds = predictions[frame_number]['rois']
                    for i in range(len(bounds)):
                        object_id = self._get_next_object_id()
                        box_data = {
                            'xtl': float(bounds[i][1]),
                            'ytl': float(bounds[i][0]),
                            'xbr': float(bounds[i][3]),
                            'ybr': float(bounds[i][2]),
                            'confidence': float(predictions[frame_number]['scores'][i])
                        }
                        class_name = self.class_names[predictions[frame_number]['class_ids'][i]]
                        if (frame_number, class_name) not in self.predictions:
                            self.predictions[(frame_number, class_name)] = {}
                        self.predictions[(frame_number, class_name)][object_id] = deepcopy(box_data)
                else:
                    print("frame {} not in predictions".format(frame_number))

        elif prediction_format == 'supervisely':
            # TODO the below assumes that filenames will always have the naming structure 'frame_XXXXX.png.json'
            for file_name in os.listdir(predictions):
                frame_id = file_name[6:11]
                frame_number = float(frame_id)
                if frame_number in self.frame_numbers:
                    frame_preds = json.load(open(os.path.join(predictions, "frame_"+frame_id+".png.json")))
                    for object in frame_preds['objects']:
                        object_id = self._get_next_object_id()
                        box_data = {
                            'xtl': float(object['points']['exterior'][0][0]),
                            'ytl': float(object['points']['exterior'][0][1]),
                            'xbr': float(object['points']['exterior'][1][0]),
                            'ybr': float(object['points']['exterior'][1][1]),
                            'confidence': float(object['tags'][0]['value'])
                        }
                        class_name = object['classTitle'].lower()
                        if (frame_number, class_name) not in self.predictions:
                            self.predictions[(frame_number, class_name)] = {}
                        self.predictions[(frame_number, class_name)][object_id] = deepcopy(box_data)

        elif prediction_format == 'zklab_measurement':
            self.predictions = predictions
        else:
            raise NotImplementedError

    def _compute_iou(self, object_a, object_b):
        """Compute the Jaccard index, defined as intersection / union (also called IoU),
        i.e. the percentage of all of the space taken up by the 2 boxes that is overlapping

        :param object_a: dict of object data including 'xtl', 'ytl', 'xbr', 'ybr' (float)
        :param object_b: dict of object data including 'xtl', 'ytl', 'xbr', 'ybr' (float)
        :return: float
        """
        # intersection area
        x_overlap = min(object_a['xbr'], object_b['xbr']) - max(object_a['xtl'], object_b['xtl'])
        y_overlap = min(object_a['ybr'], object_b['ybr']) - max(object_a['ytl'], object_b['ytl'])
        intersection = max(x_overlap, 0) * max(y_overlap, 0)

        # sum of the areas of both boxes
        a_area = (object_a['xbr'] - object_a['xtl']) * (object_a['ybr'] - object_a['ytl'])
        b_area = (object_b['xbr'] - object_b['xtl']) * (object_b['ybr'] - object_b['ytl'])
        area_sum = a_area + b_area
        assert area_sum > 0

        # intersection == total_area means perfect overlap
        if area_sum == intersection:
            return 1.0
        else:
            return intersection / (area_sum - intersection)

    def _match_objects(self, iou_threshold):
        """For each frame and each class type, use IoU to minimize the total
        object-hypothesis distance error and thereby determine the true positives
        # TODO this is consistent w/ our MOT evaluation, but different from COCO method:
        # TODO (cont'd) https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        # TODO (cont'd) Zhengye is implementing greedy matching for multicut so that will be similar to COCO, we
        # TODO (cont'd) should give it a try in here to confirm how different it is from global cost minimization

        :param iou_threshold: float value denoting minimum IoU threshold required to allow for a TP identification
        :return: does not return; updates class matches and total_matches attributes to track matched objects
        """
        matches = {}  # dict of pred to actual
        distances = {}  # dict of frame number to 2d array of pairwise distances
        self.matches[iou_threshold] = {}  # dict of pairwise IoUs

        for frame_class in self.actuals:
            if frame_class in self.predictions:
                actuals = self.actuals[frame_class]
                preds = self.predictions[frame_class]

                # build 2d matrix of distances between predicted and actual objects
                distances[frame_class] = []
                for a in actuals:
                    distance_list = []
                    for b in preds:
                        iou = self._compute_iou(actuals[a], preds[b])
                        distance = (1 - iou) if iou >= iou_threshold else self.default_high
                        distance_list.append(distance)
                    distances[frame_class].append(distance_list)

                # minimize total object-hypothesis distance error for available actual and predicted objects
                if distances[frame_class]:
                    row_indices, col_indices = linear_sum_assignment(distances[frame_class])

                    # assign matches
                    indices = list(zip(list(row_indices), list(col_indices)))
                    for row, column in indices:
                        actual = list(actuals.keys())[row]
                        pred = list(preds.keys())[column]
                        matches[pred] = actual
                        iou = 1 - distances[frame_class][row][column]
                        if iou >= iou_threshold:
                            if frame_class not in self.matches[iou_threshold]:
                                self.matches[iou_threshold][frame_class] = {}
                            self.matches[iou_threshold][frame_class][(actual, pred)] = iou
                            self.results_lists.append([iou_threshold, frame_class[0], frame_class[1], actual, pred,
                                                   preds[pred]['confidence'], 1, 0, 0])

    def _compute_errors(self, iou_threshold):
        """Count false positives and false negatives (misses)
        - any actual not matched is missed
        - any prediction not matched is a false positive

        :param iou_threshold: float value denoting minimum IoU threshold required to allow for a TP identification
        :return: does not return; stores lists of object_id ints in class attributes
        """
        # TODO inefficiently written and likely could be combined w/ self._match_objects()

        frame_classes = set(self.actuals.keys()).union(set(self.predictions.keys()))
        self.misses[iou_threshold] = {}
        self.false_positives[iou_threshold] = {}
        results_df = pd.DataFrame()

        def build_frame(frame_class):
            frame_class_results_df = pd.DataFrame()
            frame = frame_class[0]
            misses = {frame_class: []}
            false_positives = {frame_class: []}

            # loop over all object ids that we need to consider and
            # determine which of the buckets each falls into
            actuals = self.actuals[frame_class] if frame_class in self.actuals else {}
            preds = self.predictions[frame_class] if frame_class in self.predictions else {}
            # false negative if an actual is unmatched
            for a in actuals:
                if (frame_class not in self.matches[iou_threshold]) or \
                        (a not in [x[0] for x in self.matches[iou_threshold][frame_class]]):
                    misses[frame_class].append(a)
                    frame_class_results_df = frame_class_results_df.append(pd.DataFrame({
                        'iou_threshold': iou_threshold,
                        'frame_number': frame,
                        'class_name': frame_class[1],
                        'actual': a,
                        'detection': np.nan,
                        'confidence': np.nan,
                        'TP': 0,
                        'FP': 0,
                        'FN': 1
                    }, index=[0])).reset_index(drop=True)
            # false positive if a prediction is unmatched
            for b in preds:
                if (frame_class not in self.matches[iou_threshold]) or \
                        (b not in [x[1] for x in self.matches[iou_threshold][frame_class]]):
                    false_positives[frame_class].append(b)
                    frame_class_results_df = frame_class_results_df.append(pd.DataFrame({
                        'iou_threshold': iou_threshold,
                        'frame_number': frame,
                        'class_name': frame_class[1],
                        'actual': np.nan,
                        'detection': b,
                        'confidence': preds[b]['confidence'],
                        'TP': 0,
                        'FP': 1,
                        'FN': 0
                    }, index=[0])).reset_index(drop=True)
            return frame_class_results_df, misses, false_positives

        results = parmap(build_frame, frame_classes)
        for frame_class_results in results:
            results_df = results_df.append(frame_class_results[0])
            self.misses[iou_threshold].update(frame_class_results[1])
            self.false_positives[iou_threshold].update(frame_class_results[2])
        self.results_df = self.results_df.append(results_df)

    def get_n_objects(self, iou_threshold, frame_number=None):
        """Return the number of ground truth objects, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to int (number of objects in the video or frame)
        """
        actuals_df = deepcopy(self.results_df[~np.isnan(self.results_df['actual'])])
        return_dict = {}
        if frame_number:
            for class_name in self.class_names[1:]:  # skip 'bg'
                return_dict[class_name] = len(actuals_df[(actuals_df['iou_threshold'] == iou_threshold) & (actuals_df['frame_number']==frame_number) & (actuals_df['class_name'] == class_name)])
            return_dict['total'] = len(actuals_df[(actuals_df['iou_threshold']==iou_threshold) & (actuals_df['frame_number']==frame_number)])
        else:
            for class_name in self.class_names[1:]:  # skip 'bg'
                return_dict[class_name] = len(actuals_df[(actuals_df['iou_threshold'] == iou_threshold) & (actuals_df['class_name'] == class_name)])
            return_dict['total'] = len(actuals_df[(actuals_df['iou_threshold'] == iou_threshold)])
        return return_dict

    def get_n_matches(self, iou_threshold, frame_number=None):
        """Find the count of matches made at a given IoU threshold
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class name) to float (mean IoU)
        """
        frame_numbers = [frame_number] if frame_number else self.frame_numbers
        matches = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            match_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.matches[iou_threshold]:
                    match_count += len(self.matches[iou_threshold][frame_class])
            matches[class_name] = match_count
            total_count += match_count
        matches['total'] = total_count
        return matches

    def get_n_misses(self, iou_threshold, frame_number=None):
        """Return the count of objects missed by the tracker, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to float
        """
        frame_numbers = [frame_number] if frame_number else self.frame_numbers
        misses = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            miss_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.misses[iou_threshold]:
                    miss_count += len(self.misses[iou_threshold][frame_class])
            misses[class_name] = miss_count
            total_count += miss_count
        misses['total'] = total_count
        return misses

    def get_n_false_positives(self, iou_threshold, frame_number=None):
        """Return the count of objects mistakenly identified by the tracker
        that do not actually exist at all, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to float
        """
        frame_numbers = [frame_number] if frame_number else self.frame_numbers
        false_positives = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            fp_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.false_positives[iou_threshold]:
                    fp_count += len(self.false_positives[iou_threshold][frame_class])
            false_positives[class_name] = fp_count
            total_count += fp_count
        false_positives['total'] = total_count
        return false_positives

    def get_average_precision(self, class_name, recall_levels=None, iou_threshold=None, frame_number=None, smooth=True):
        """Calculate the average precision (true positives
        divided by all predictions) for a given object class,
        using a given iou_threshold, across a set of recall levels
        Optionally for a single frame (defaults to computing across all frames)

        :param class_name: string name of object class
        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_threshold: float value of minimum valid IoU; if None, defaults to self.primary_iou_threshold
                              must be one of the values passed to the constructor (either for iou_thresholds or
                              primary_iou_threshold) or else matches and errors will not be available
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return float
        """
        recall_levels = recall_levels if recall_levels else self.recall_levels
        iou_threshold = iou_threshold if iou_threshold else self.primary_iou_threshold
        if iou_threshold not in self.iou_thresholds:
            raise ValueError("iou_threshold must be one of the values passed into the class constructor")
        results = deepcopy(self.results_df)
        results = results.loc[(results['class_name'] == class_name) & (results['iou_threshold'] == iou_threshold)]

        if frame_number:
            results = results.loc[results['frame_number']==frame_number]

        # rank by confidence and compute recall moving downward
        if results.empty:
            return np.nan
        else:
            results = results[results['detection'].notnull()].sort_values(by='confidence', ascending=False)
            recalls = []
            true_positives = 0
            false_positives = 0
            for i, row in results.iterrows():
                true_positives += row['TP']
                false_positives += row['FP']
                recall = true_positives / (true_positives + false_positives)
                recalls.append(recall)
            results['recall'] = np.asarray(recalls)

            # compute average precision across recall levels
            precision_list = []
            for recall in recall_levels:
                if len(results.loc[(results['recall'] >= recall)]) > 0:
                    precision = sum(results.loc[(results['recall'] >= recall)]['TP']) / \
                                len(results.loc[(results['recall'] >= recall)])
                    precision_list.append(precision)

            if smooth:
                # ensure monotonicity
                last_low_idx = 0
                for i in range(len(precision_list)):
                    if precision_list[i] > precision_list[last_low_idx]:
                        while last_low_idx < i:
                            precision_list[last_low_idx] = precision_list[i]
                            last_low_idx += 1
                    else:
                        last_low_idx = i

            return sum(precision_list) / len(precision_list) if len(precision_list) else 0

    def get_all_average_precisions(self, recall_levels=None, iou_thresholds=None, frame_number=None):
        """Compute the average precision for each class and across
        all classes, for all IoU thresholds for which we have data
        Optionally for a single frame (defaults to computing across all frames)

        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_thresholds: list of float; if None, defaults to self.iou_thresholds
        :param frame_number: None or float, frame ID number
        :return: dict of string (class name) to float (avg. precision)
        """
        class_precisions = {}
        total_sum = 0
        total_count = 0
        recall_levels = recall_levels if recall_levels else self.recall_levels
        iou_thresholds = iou_thresholds if iou_thresholds else self.iou_thresholds
        for class_name in self.class_names[1:]:  # skip 'bg'
            precision = 0
            count = 0
            for iou_threshold in iou_thresholds:
                ct_precision = self.get_average_precision(class_name, recall_levels, iou_threshold, frame_number)
                if not np.isnan(ct_precision):
                    precision += ct_precision
                    count += 1
            class_precisions[class_name] = precision / count if count else 0
            total_sum += precision
            total_count += count
        class_precisions['total'] = total_sum / total_count if total_count else 0
        return class_precisions

    def get_mean_average_precision(self, recall_levels=None, iou_thresholds=None, frame_number=None):
        """Compute the mean of the average precision across all
        classes and all IoU thresholds for which we have data
        Optionally for a single frame (defaults to computing across all frames)

        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_thresholds: list of float; if None, defaults to self.iou_thresholds
        :param frame_number: None or float, frame ID number
        :return: float
        """
        return self.get_all_average_precisions(recall_levels, iou_thresholds, frame_number=None)['total']

    def get_all_ious(self, iou_threshold, frame_number=None):
        """Find the avg. IoU at a given IoU threshold
        for each class and across all classes
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :return: dict of string (class name) to float (mean IoU)
        """
        frame_numbers = [frame_number] if frame_number else self.frame_numbers
        class_ious = {}
        total_sum = 0
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            iou_sum = 0
            iou_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.matches[iou_threshold]:
                    for pair, iou in self.matches[iou_threshold][frame_class].items():
                        iou_sum += iou
                        iou_count += 1
            class_ious[class_name] = iou_sum / iou_count if iou_count else 0
            total_sum += iou_sum
            total_count += iou_count
        class_ious['total'] = total_sum / total_count if total_count else 0
        return class_ious

    def get_all_metrics(self, frame_number=None, return_format='json'):
        """Return all intermediate and final metrics, at either a given time or over all frames

        :param frame_number: None or float, frame ID number
        :param return_format: String, one of ['json', 'df']
        :return: dict of string (metric name) to float (metric value)
        """
        return_dict = {
            'actual_boxes': self.get_n_objects(iou_threshold=self.primary_iou_threshold, frame_number=frame_number),
            'matches': self.get_n_matches(iou_threshold=self.primary_iou_threshold, frame_number=frame_number),
            'misses': self.get_n_misses(iou_threshold=self.primary_iou_threshold, frame_number=frame_number),
            'false_positives': self.get_n_false_positives(iou_threshold=self.primary_iou_threshold, frame_number=frame_number),
            'mIoU': self.get_all_ious(iou_threshold=self.primary_iou_threshold, frame_number=frame_number),
            'AP': self.get_all_average_precisions(iou_thresholds=[self.primary_iou_threshold], frame_number=frame_number),
            'mAP': self.get_mean_average_precision(frame_number=frame_number)
        }
        return return_dict if return_format == 'json' else pd.DataFrame.from_dict(return_dict)
