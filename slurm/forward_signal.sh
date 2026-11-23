# Sourced by env.sh. Defines run_forwarding_usr1, which every training job uses to launch
# tools/train.py so that the walltime warning actually reaches it.
#
# WHY THIS EXISTS
# ---------------
# The training chains survive the 72-hour walltime by having Slurm send SIGUSR1 ten minutes
# early: train.py saves at the exact step and exits, and the batch script then submits the
# follow-up job. That only works if the signal reaches train.py. Plain `--signal=USR1@600`
# signals job *steps* -- processes started with srun -- and train.py is not one: it is a child
# of the batch shell. So nothing received the warning; at 72 hours Slurm killed the job, the
# batch shell died with it, and the follow-up was never submitted. Every chain whose link
# reached the walltime ended right there, silently: no_uncertainty and fusion_sum on Nov 19,
# no_instance_path on Nov 23. Only links that crashed first were requeued, by luck.
#
# The scripts now request `--signal=B:USR1@600`, which goes to the batch shell. A shell
# waiting on a foreground child does not run its traps until the child exits, so the child
# is started in the background and the shell `wait`s on it: `wait` returns early when a
# trapped signal arrives, the trap forwards SIGUSR1, and the loop waits again for the child
# to finish saving. The exit status returned is the child's own, crashes (139) included.
#
# Only for a single process. torchrun treats SIGUSR1 as a shutdown signal rather than
# forwarding it to its workers, so multi-GPU runs do not get the clean save.

run_forwarding_usr1() {
    "$@" &
    local pid=$!
    # shellcheck disable=SC2064  # expand $pid now, when the trap is set
    trap "echo '[slurm] walltime warning received -- forwarding SIGUSR1 to pid $pid'; kill -USR1 $pid 2>/dev/null || true" USR1
    local rc=0
    wait "$pid" || rc=$?
    # A trapped signal makes `wait` return early (status > 128) while the child is still
    # running and saving; wait again until it is really gone.
    while kill -0 "$pid" 2>/dev/null; do
        rc=0
        wait "$pid" || rc=$?
    done
    trap - USR1
    return "$rc"
}
