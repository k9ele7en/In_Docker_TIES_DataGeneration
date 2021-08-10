# original filename
file_name=gentable.log
 
# assign timestamp to filename
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
echo "Current Time : $current_time"

new_fileName=$current_time.$file_name
 
cp $file_name $new_fileName

date 2>&1 | tee -a $new_fileName

# update flags before run
python generate_data.py --filesize 4 --threads 2 --imagespath UNLV_dataset/unlv_images --ocrpath UNLV_dataset/unlv_xml_ocr --tablepath UNLV_dataset/unlv_xml_ocr --visualizeimgs 1 --visualizebboxes 1 2>&1 | tee -a $new_fileName

date 2>&1 | tee -a $new_fileName
