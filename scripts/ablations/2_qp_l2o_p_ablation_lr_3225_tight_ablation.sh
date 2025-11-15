# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.0000001 \
#     --e 1 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir ablation_fullsparse_sumloss_T100/SGDlr0.0000001-T100-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-RandX0plus100


# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.000001 \
#     --e 1 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir ablation_fullsparse_sumloss_T100/SGDlr0.000001-T100-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-RandX0plus100



# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.00001 \
#     --e 1 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir ablation_fullsparse_sumloss_T100/SGDlr0.00001-T100-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-RandX0plus100



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 32 \
    --output-dim 25 \
    --lstm-hidden-size 1024 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0001 \
    --e 1 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss_T100/SGDlr0.0001-T100-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-RandX0plus100



# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.001 \
#     --e 1 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir ablation_fullsparse_sumloss_T100/SGDlr0.001-T100-rand1orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-RandX0plus100



# sh scripts/2_qp_l2o_p_ablation_lr_6450.sh
