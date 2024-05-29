for suffix in {300..310};
do
  for n_guide_steps in 1 2;
  do
    for t_stopgrad in 2 4;
    do
      for scale in 0.1 0.01 0.001 0.0001;
      do
        python scripts/hl_plan_guided.py \
          --dataset halfcheetah-medium-expert-v2 \
          --suffix $suffix \
          --n_guide_steps $n_guide_steps \
          --scale $scale \
          --t_stopgrad $t_stopgrad
      done
    done
  done
done
