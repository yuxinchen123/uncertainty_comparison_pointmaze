Context
/p/rlprojects/RND/07_reconstruction

We implement a PointMaze env and an RND algorithm in SAC
Now I want to shift the algorithm to PPO, the base implementation to use is
https://github.com/vwxyzjn/cleanrl (find the ppo+rnd) implemenation
For this PPO,

I want two update style, 
one is update on the whole batch of the data them discard the data (update on the 512 samples the discard all of them)

another one is 
for epoch in range(4):
    shuffle(all_512_samples)

    for minibatch in 4_minibatches:
        # minibatch contains 128 samples

        compute_loss(minibatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       # <-- UPDATE HERE

you might need to fuse them separately, (also check if we only present the on-policy one in the code would improve performance, because i am afraible an if sentence would break the fusion)

You are an expert AI infra engineer
Now I want to improve the efficiency of the training 

I am going to tell you a few module you need to improvement, for each model you need to follow the style of 
https://github.com/karpathy/autoresearch
clone it locally, so you have a reference
to loop unitil you get the best performance 

for each module have a document to track what you have changed, and whether it gives good or bad impact.

and loop until the performance is great. 

Module 1:
Improve the environment performance, 
I want the pointmaze env to be extremely efficienct, 
since we are going to train on GPU, 
I want the env to be fully parallelized on GPU without any cpu interaction or minimal cpu interaction, this helps us to parallelize thousands or even more env on GPU.

I want 3 versions of env
(1) PyTorch version 
(2) fused CUDA-level version 
(3) jax version

for each module, you run the 
https://github.com/karpathy/autoresearch
to loop until you get the best performance 

you keep a progress file for each of them separately, 
although it would be a good idea to cross-reference to get experience.


Module 2:
After we have done module 1:
the next is to improve the architecture performance, which is the PPO 
when you run this, you might also need an env, in this case, use the best version you optimized in Module 1.


Note, one thing I need to modify the code is, I want to train multiple copies of RL, and each one have their env and network
That means, the env has to be able to batch multiple copies 
and the network has to batch multiple copies and so on 
Essentially, I want it to be efficient; we are training 100 independent seeds at the same time on a GPU.  I want the number of copies to be a knob I can adjust.

This requires the modification of both the env and the PPO algorithm.

I don't need much log of the middle step, remove the wandb or tensor board log and free feel to make the log sparse, for example 20 mins a log



You should search new papers and famous GitHub repos such as vLLM and sgLang for this part, 
also make sure you have tried every performance trick mentioned in 
/p/rlprojects/RLforOR/LLMs-from-scratch 
https://github.com/karpathy/nanoGPT (clone it locally to this /p/rlprojects/RND/09_parallelization/reference_repo)
https://github.com/karpathy/nanochat (clone it locally to this /p/rlprojects/RND/09_parallelization/reference_repo)

for this module, I want 
(1) a pytorch module 
(2) a JAX module 




Module 3 (end-to-end training)
Connecting Module 1 and Module 2:
It might be possible to stack module 1 above module 2 so they looks like a unified network 
For this part, I want you optimize the 
module 1 (best of jax, pytorch or fused cuda) + module 2 (pytorch)
module 1 (best of jax, pytorch or fused cuda) + module 2 (jax)
though you should try each of module 1 here because some of them might be easily fused (my guess try each of them, improve each of them with the training architecture, and verify it)

Extra step:
After you finish the auto research 
go back to module 1
look my 
/p/rlprojects/RLforOR/inventory_management/joint_replenishment/efficiency
and check if you have any missed techniques (including infra harness and optimization techniques for each module), 
then improve modules 1, 2, and 3 again (but don't look at this repo at first, only at this step)




You maintain a separate progress_and_changes.md
for each subtask in each of the modules 

design the folder structure first before you start, 
/p/rlprojects/RND/09_parallelization

for everything, use the serval 05 direct ssh, 
at the same time, only run 1 profiling on it, so you don't get crowding issue 

Don't rush; you have 3 days to complete this tasks. 
However, don't stop and ask me, because i am leaving, you will stuck and not make progress if you wait my response. 



After you have finish all of them, 

You final dievelry is to train 8, 16, 32, 64, and 128. (indepdent RND runs, simialr to 32 seeds, until the GPU can't fit it)
all pytorch version 
show the 
env throughput
training throughput 
end-to-end throughput
before and after update in a table 
show the performance table 






moreover, also profile the time spent on each part in a table with time and percentage 
for example
env sampling (break down into sampling, bottleneck code operations, and so on, so I know what the bottleneck of the systems)
training pipeline (simialrly breakdown into critical blocks)


then show the throughput of every 
Module 1 
(1) PyTorch version 
(2) fused CUDA-level version 
(3) jax version

By showing 
(1) how many number of env can it paralized in H100 in a line, and show me after what number it begins to increase the time, this line should be in log scale
(2) the speed of the step per env 

Module 2 traiing
Then, show the throughput of every 
(1) a pytorch module 
(2) a jax module 
(you decide which env the C++ will be paired with)

to get the best performance, disperse a lot of subagents to review your approach 
and a lot of subgaents to search and propose ideas 
and a lot of subagents to read https://github.com/karpathy/autoresearch (clone it locally to this /p/rlprojects/RND/09_parallelization/reference_repo)
between substeps and steps.

Moedule 3 end to end
(1) pytorch 
(2) jax


Moreover, also report the number of batch training copies 
including 
the total allow of batch training copies, 
and whether batch training copies increase would decrease the per training copy through put or not in a table and a graph 
and whether batch training copies increase would decrease the total training through put or not in a table and a graph 

Feel free to use fable subagent if needed. 
for each subagent you can let them work in a branch following
/p/rlprojects/RLforOR/parallel-agents-skill.zip (unzip it, and you can modify it to make it suitable for this tasks)
but remember, only one training should run on the H100 at the same time, if agent A is using then agent B should wait the lock release then use the H100



frequently review my prompt (saved in /p/rlprojects/RND/09_parallelization/efficiency_improvement_user_prompt.md) 
and  https://github.com/karpathy/autoresearch whenever you don't know what to do next

whenevery you start a trianig, set a 20 mins session cron to monitor.

I want one unified report in the end 
/p/rlprojects/RND/09_parallelization/report/<date>-name

you can create new virtual env if needed