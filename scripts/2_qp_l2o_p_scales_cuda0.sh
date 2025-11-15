
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 32 \
#     --output-dim 20 \
#     --lstm-hidden-size 1024 \
#     --init-lr 0.0000001 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --save-dir SGDlr0.0000001-T100-rand100orth-3220-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --init-lr 0.0000001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --save-dir SGDlr0.0000001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 128 \
    --output-dim 100 \
    --lstm-hidden-size 4096 \
    --init-lr 0.00001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --save-dir SGDlr0.0000001-T100-rand100orth-128100-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 256 \
    --output-dim 200 \
    --lstm-hidden-size 8192 \
    --train-batch-size 24 \
    --val-size 24 \
    --init-lr 0.0000001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --save-dir SGDlr0.0000001-T100-rand100orth-256200-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0
    