python -u as_learn.py \
    --model_path google/pegasus-large --dataset_name xsum \
    --output_path as_bayes_pegasus_e10_st75_k500_l50 \
    --validation_file xsum_data/validation.json \
    --text_column document \
    --summary_column summary \
    --seed 999 \
    --resume 0 \
    --acquisition "bayesian" --steps 75 \
    --N 10 --L 50 --K 500 --S 10 \
    --init_model google/pegasus-large \
    --max_source_length 256 --max_summary_length 62 \
    --batch_size 6 \
    --num_beams 3 \
    --max_val_samples 100 \
    --max_test_samples 5000 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --save_step 100 --save_limit 1 \
    --metric_for_best_model rouge1