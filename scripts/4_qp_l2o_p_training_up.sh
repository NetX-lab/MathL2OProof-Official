
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --lstm-hidden-size 20 \
#     --train-batch-size 2048 \
#     --val-size 2048 \
#     --init-lr 0.001 \
#     --e 100 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --identical-dict \
#     --fixed-dict \
#     --save-dir training/FixedM/SGDlr0.001-T100-rand100orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-UP


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 20 \
    --train-batch-size 2048 \
    --val-size 2048 \
    --init-lr 0.001 \
    --e 100 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --identical-dict \
    --save-dir training/MultipleM/SGDlr0.001-T100-rand100orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-UP