# Simple 40M example
python cs336_basics/train/train.py --vocab-size 10000 \
 --context-length 256 \
 --d-model 64 \
 --d-ff 256 \
 --rope-theta 10000 \
 --num-layers 4 \
 --num-heads 4 \
 --num-train-tokens 40000000 \
 --max-lr 0.0001 \
 --min-lr 0 \
 --lr-scheduler-type cosine \
 --beta1 0.9 \
 --beta2 0.999 \
 --train-path tokenized/tinystories-train.npy \
 --validation-path tokenized/tinystories-valid.npy \
 --batch-size 32 \
 --dtype torch.float32