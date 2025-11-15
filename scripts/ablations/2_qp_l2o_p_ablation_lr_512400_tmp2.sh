python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.0000001 \
    --e 300 \
    --optimizer-training-steps 20    --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.0000001-T20-rand300orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.000001 \
    --e 300 \
    --optimizer-training-steps 20    --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.000001-T20-rand300orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.00001 \
    --e 300 \
    --optimizer-training-steps 20    --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand300orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.0001 \
    --e 300 \
    --optimizer-training-steps 20    --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.0001-T20-rand300orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 300 \
    --optimizer-training-steps 20    --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.001-T20-rand300orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


