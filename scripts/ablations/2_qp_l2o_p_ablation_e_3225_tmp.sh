python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 100 \
    --optimizer-training-steps 20   --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand100orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 50 \
    --optimizer-training-steps 20   --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand50orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 25 \
    --optimizer-training-steps 20   --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand25orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 5 \
    --optimizer-training-steps 20   --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand5orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.00001 \
    --e 1 \
    --optimizer-training-steps 20   --unroll-length 20 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T20/SGDlr0.00001-T20-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


# sh scripts/2_qp_l2o_p_ablation_e_6450.sh