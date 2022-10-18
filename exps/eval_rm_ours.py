#!/usr/bin/env python3

import fire

from summarize_from_feedback import eval_rm_ours
from summarize_from_feedback.utils import experiment_helpers as utils
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher


def experiment_definitions():

    rmours = combos(
        bind_nested("task", utils.tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", utils.rm4()),
        bind("input_path", "./"),
    )

    return locals()


if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=eval_rm_ours.main, mode="local"
        )
    )
