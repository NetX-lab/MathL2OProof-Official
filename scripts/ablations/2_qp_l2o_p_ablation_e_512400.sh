python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 2000 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T10-rand2000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 3000 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T10-rand3000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 5000 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T10-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 10000 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T10-rand10000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 20000 \
    --optimizer-training-steps 10    --unroll-length 10 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T10/SGDlr0.001-T10-rand20000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


