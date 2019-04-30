python3 train.py \
--dataset_mode texture \
--name expression-transfer \
--preprocess resize \
--load_size 256 \
--data-class-a ~/Projects-2017-11/emotion-transfer/test/data/common-1/images/texture-pat-dlib-512/common_part3_neutral \
--data-class-b ~/Projects-2017-11/emotion-transfer/test/data/common-1/images/texture-pat-dlib-512/common_part1_middle_smile \
--continue_train \
