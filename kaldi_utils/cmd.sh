# you can change cmd.sh depending on what type of run you are using.
# If you have no runing system and want to run on a local machine, you
# can change all instances 'run.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  run.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different runs are configured differently, with different
# run names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/run.conf to match your queue's configuration.  Search for
# conf/run.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/run.pl or utils/slurm.pl.

export train_cmd=run.pl
export decode_cmd="run.pl --mem 2G"
# the use of cuda_cmd is deprecated, used only in 'nnet1',
export cuda_cmd="run.pl --gpu 1"
