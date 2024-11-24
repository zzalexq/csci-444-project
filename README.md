# How to Generate Alignments Using Awesome Align
DATA_FILE=./zhen.src-tgt
MODEL_NAME_OR_PATH=./model_without_co
OUTPUT_FILE=./output.txt
OUTPUT_WORDS=./alignments.txt
OUTPUT_PROB=./alignments-prob.txt

CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32 \
    --num_workers 0 \
    --output_word_file=$OUTPUT_WORDS \
    --output_prob_file=$OUTPUT_PROB 


WORK TO BE DONE:
- [ ] TEST UNMODIFIED MBART ON TEXT DATA
- [x] PROCESS TEXT DATA FOR WORD ALIGNMENTS AND PAIRS
- [ ] IDENTIFY ANCHOR POINT WORDS TO USE FROM ALIGNMENTS 
- [ ] MODIFY MBART TO USE ANCHOR POINTS 
- [ ] TEST MODIFIED MBART ON TEXT DATA
- [ ] INCORPORATE RL 
- [ ] TEST TEXT DATA WITH RL MODEL

