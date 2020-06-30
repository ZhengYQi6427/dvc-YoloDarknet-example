#!/bin/sh
# sample scripting

##### CONSTANT

#echo -n "Enter your github repository url > "
#read github_link
echo -n "Enter your Github repository name > "
read repo_name
#echo -n "Enter your Synology video file path > "
#read video_link
NAS_account="ra_yz3622"
NAS_password="summer"
#echo -n "Enter yout target directory > "
#read dir

##### FUNCTIONS

get_videos()
{
   git clone https://github.com/ZhengYQi6427/DVC-data.git
   mv $(pwd)/DVC-data/images $(pwd)
   mv $(pwd)/DVC-data/annotations $(pwd)
   rm -rf DVC-data
   
   dvc add $(pwd)/images
   dvc add $(pwd)/annotations
   git add images.dvc
   git add annotations.dvc
   git commit -m 'add data archive'
}

get_code()
{
   git clone "$github_link"
   rm -rf .git
   cd $repo_name
   sed -i 's+$(pwd)+g' json2voc.py
   cd ..

   git add $(pwd)/$repo_name
   git commit -m 'add code'
}

init()
{  
   if [ -e $dir ]; then
       cd $dir
       echo "Successfully open directory $dir"
   else
       echo "No directory named $dir"
       echo -n "Generate a new directory $dir? [y/n]: "
       read
       if [ "$REPLY" = "y" ]; then
          mkdir $dir
          cd $dir
       else
          echo "Sorry, please create a directory for git and dvc first!"
          exit 0
       fi
   fi

   echo -n "Do you want to install or update DVC? [y/n]"
   read
   if [ "$REPLY" = "y" ]; then 
       sudo apt update
       sudo apt install dvc
       sudo apt install tree
   fi
 
   echo -n "Enter git user name > "
   read git_user_name
   echo -n "Enter git user email > "
   read git_user_email

   git init
   dvc init
}

preprocess(){
    echo -n "Begin preprocessing? [y/n] "
    read
    if [ "$REPLY" = "y" ]; then
        echo -n "How many preprocessing scripts need to run > "
        read file_num

        cd $repo_name
        sudo apt install python-pip
        pip install -r requirements.txt
        echo "$(pwd)"

        i=0
        until [ $i -ge $file_num ];
        do
            echo -n "Enter file $i name > "
            read file_name
            if [ !-e $file_name ]; then
                echo "ERROR: File $varname doesn't exist"
                exit 0
            fi
            python $file_name
            i=$((i + 1))
        done
    fi
}

##### MAIN

# init
# rm -rf .dvc
# rm -rf .git
# get_videos
# get_code
preprocess
