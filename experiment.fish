python3 train.py \
--dataset_mode texture \
--name expression-transfer-x400 \
--preprocess resize \
--load_size 400 \
--save_epoch_freq 1 \
--data-class-a ../data/common-1/images/texture-pat-dlib-512/common_part3_neutral \
--data-class-b ../data/common-1/images/texture-pat-dlib-512/common_part1_middle_smile \
--continue_train \
