python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0000001 \
    --e 100 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.0000001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.000001 \
    --e 100 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.000001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 100 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.00001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0001 \
    --e 100 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.0001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.001 \
    --e 100 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


