import os
import os.path

import hypothesis.strategies as st
from hypothesis import assume

from ert._c_wrappers.enkf import ConfigKeys
from ert._c_wrappers.job_queue import QueueDriverEnum
from ert._c_wrappers.util.enums import MessageLevelEnum

from .egrid_generator import egrids

words = st.text(
    min_size=4, max_size=8, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
)


@st.composite
def define_keys(draw):
    """
    When initializing from dict, unlike when reading from file,
    defines are not resolved so we must make sure a define key
    is not present in any file names etc. Therefore all file_names
    are upper case and define keys are prefixed with the lower case
    'key'.
    """
    return "key" + draw(words)


def touch(filename):
    with open(filename, "w") as fh:
        fh.write(" ")


file_names = words


@st.composite
def directory_names(draw):
    return "dir" + draw(words)


small_floats = st.floats(min_value=1.0, max_value=10.0, allow_nan=False)
positives = st.integers(min_value=1, max_value=10000)

log_levels = st.sampled_from(
    [
        MessageLevelEnum.LOG_CRITICAL,
        MessageLevelEnum.LOG_ERROR,
        MessageLevelEnum.LOG_WARNING,
        MessageLevelEnum.LOG_INFO,
        MessageLevelEnum.LOG_DEBUG,
    ]
)

queue_systems = st.sampled_from(
    [
        QueueDriverEnum.LSF_DRIVER,
        QueueDriverEnum.LOCAL_DRIVER,
        QueueDriverEnum.RSH_DRIVER,
        QueueDriverEnum.TORQUE_DRIVER,
        QueueDriverEnum.SLURM_DRIVER,
    ]
)


def valid_queue_options(queue_system):
    valids = [ConfigKeys.MAX_RUNNING]
    if queue_system == QueueDriverEnum.LSF_DRIVER:
        valids += [
            "LSF_RESOURCE",
            "LSF_SERVER",
            "LSF_QUEUE",
            "LSF_LOGIN_SHELL",
            "LSF_RSH_CMD",
            "BSUB_CMD",
            "BJOBS_CMD",
            "BKILL_CMD",
            "BHIST_CMD",
            "BJOBS_TIMEOUT",
            "DEBUG_OUTPUT",
            "EXCLUDE_HOST",
            "PROJECT_CODE",
            "SUBMIT_SLEEP",
        ]
    elif queue_system == QueueDriverEnum.RSH_DRIVER:
        valids += [
            "RSH_HOSTLIST",
            "RSH_CLEAR_HOSTLIST",
            "RSH_CMD",
        ]
    elif queue_system == QueueDriverEnum.SLURM_DRIVER:
        valids += [
            ConfigKeys.SLURM_SBATCH_OPTION,
            ConfigKeys.SLURM_SCANCEL_OPTION,
            ConfigKeys.SLURM_SCONTROL_OPTION,
            ConfigKeys.SLURM_MEMORY_OPTION,
            ConfigKeys.SLURM_MEMORY_PER_CPU_OPTION,
            ConfigKeys.SLURM_EXCLUDE_HOST_OPTION,
            ConfigKeys.SLURM_INCLUDE_HOST_OPTION,
            ConfigKeys.SLURM_SQUEUE_TIMEOUT_OPTION,
            ConfigKeys.SLURM_MAX_RUNTIME_OPTION,
        ]
    elif queue_system == QueueDriverEnum.TORQUE_DRIVER:
        valids += [
            "QSUB_CMD",
            "QSTAT_CMD",
            "QDEL_CMD",
            "QUEUE",
            "NUM_CPUS_PER_NODE",
            "NUM_NODES",
            "KEEP_QSUB_OUTPUT",
            "CLUSTER_LABEL",
            "JOB_PREFIX",
            "DEBUG_OUTPUT",
            "SUBMIT_SLEEP",
        ]
    return valids


def valid_queue_values(option_name):
    if option_name in [
        "QSUB_CMD",
        "QSTAT_CMD",
        "QDEL_CMD",
        "QUEUE",
        "CLUSTER_LABEL",
        "JOB_PREFIX",
        "DEBUG_OUTPUT",
        "LSF_RESOURCE",
        "LSF_SERVER",
        "LSF_QUEUE",
        "LSF_LOGIN_SHELL",
        "LSF_RSH_CMD",
        "BSUB_CMD",
        "BJOBS_CMD",
        "BKILL_CMD",
        "BHIST_CMD",
        "BJOBS_TIMEOUT",
        "EXCLUDE_HOST",
        "PROJECT_CODE",
        "RSH_HOSTLIST",
        "RSH_CLEAR_HOSTLIST",
        "RSH_CMD",
        ConfigKeys.SLURM_SBATCH_OPTION,
        ConfigKeys.SLURM_SCANCEL_OPTION,
        ConfigKeys.SLURM_SCONTROL_OPTION,
        ConfigKeys.SLURM_MEMORY_OPTION,
        ConfigKeys.SLURM_MEMORY_PER_CPU_OPTION,
        ConfigKeys.SLURM_EXCLUDE_HOST_OPTION,
        ConfigKeys.SLURM_INCLUDE_HOST_OPTION,
    ]:
        return words
    if option_name in [
        "SUBMIT_SLEEP",
        ConfigKeys.SLURM_SQUEUE_TIMEOUT_OPTION,
    ]:
        return st.builds(str, small_floats)
    if option_name in [
        "NUM_CPUS_PER_NODE",
        "NUM_NODES",
        "MAX_RUNNING",
        ConfigKeys.SLURM_MAX_RUNTIME_OPTION,
    ]:
        return st.builds(str, positives)
    if option_name in [
        "KEEP_QSUB_OUTPUT",
        "DEBUG_OUTPUT",
    ]:
        return st.builds(str, st.booleans())
    else:
        raise ValueError(f"unknown option {option_name}")


@st.composite
def queue_options(draw, queue_systems):
    queue_system = draw(queue_systems)
    name = draw(st.sampled_from(valid_queue_options(queue_system)))
    return {
        ConfigKeys.NAME: name,
        ConfigKeys.VALUE: draw(valid_queue_values(name)),
    }


@st.composite
def config_dicts(draw):
    queue_system = draw(queue_systems)
    config_dict = draw(
        st.fixed_dictionaries(
            {
                ConfigKeys.NUM_REALIZATIONS: positives,
                ConfigKeys.ECLBASE: st.just(draw(words) + "%d"),
                ConfigKeys.RUNPATH_FILE: file_names,
                ConfigKeys.ALPHA_KEY: small_floats,
                ConfigKeys.ITER_CASE: words,
                ConfigKeys.ITER_COUNT: positives,
                ConfigKeys.ITER_RETRY_COUNT: positives,
                ConfigKeys.RERUN_KEY: st.booleans(),
                ConfigKeys.RERUN_START_KEY: positives,
                ConfigKeys.UPDATE_LOG_PATH: directory_names(),
                ConfigKeys.STD_CUTOFF_KEY: small_floats,
                ConfigKeys.SINGLE_NODE_UPDATE: st.booleans(),
                ConfigKeys.MAX_RUNTIME: positives,
                ConfigKeys.MIN_REALIZATIONS: positives,
                ConfigKeys.CONFIG_DIRECTORY: st.just(os.getcwd()),
                ConfigKeys.CONFIG_FILE_KEY: file_names,
                ConfigKeys.DEFINE_KEY: st.dictionaries(define_keys(), words),
                ConfigKeys.DATA_KW_KEY: st.dictionaries(words, words),
                ConfigKeys.DATA_FILE: file_names,
                ConfigKeys.GRID: st.just(draw(words) + ".EGRID"),
                ConfigKeys.JOB_SCRIPT: file_names,
                ConfigKeys.USER_MODE: st.booleans(),
                ConfigKeys.MAX_SUBMIT: positives,
                ConfigKeys.NUM_CPU: positives,
                ConfigKeys.QUEUE_SYSTEM: st.just(queue_system),
                ConfigKeys.QUEUE_OPTION: st.lists(queue_options(st.just(queue_system))),
                ConfigKeys.ANALYSIS_COPY: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.SRC_NAME: st.just("STD_ENKF"),
                            ConfigKeys.DST_NAME: words,
                        }
                    )
                ),
                ConfigKeys.ANALYSIS_SET_VAR: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.MODULE_NAME: st.just("STD_ENKF"),
                            ConfigKeys.VAR_NAME: st.just("ENKF_NCOMP"),
                            ConfigKeys.VALUE: st.integers(),
                        }
                    )
                ),
                ConfigKeys.ANALYSIS_SELECT: st.just("STD_ENKF"),
                ConfigKeys.INSTALL_JOB: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.NAME: words,
                            ConfigKeys.PATH: file_names,
                        }
                    ),
                ),
                ConfigKeys.INSTALL_JOB_DIRECTORY: st.lists(directory_names()),
                ConfigKeys.LICENSE_PATH: directory_names(),
                ConfigKeys.UMASK: st.just(0x007),
                ConfigKeys.RANDOM_SEED: words,
                ConfigKeys.SETENV: st.lists(
                    st.fixed_dictionaries(
                        {ConfigKeys.NAME: words, ConfigKeys.VALUE: words}
                    )
                ),
            }
        )
    )
    should_exist_files = [
        job[ConfigKeys.PATH] for job in config_dict[ConfigKeys.INSTALL_JOB]
    ]
    should_exist_files.append(config_dict[ConfigKeys.DATA_FILE])
    should_exist_files.append(config_dict[ConfigKeys.JOB_SCRIPT])

    for filename in should_exist_files:
        if not os.path.isfile(filename):
            print(f"touch {filename}")
            touch(filename)

    should_exist_directories = config_dict[ConfigKeys.INSTALL_JOB_DIRECTORY]
    for dirname in should_exist_directories:
        if not os.path.isdir(dirname):
            print(f"mkdir {dirname}")
            os.mkdir(dirname)

    draw(egrids).to_file(config_dict[ConfigKeys.GRID])

    assume(config_dict[ConfigKeys.DATA_FILE] != config_dict[ConfigKeys.CONFIG_FILE_KEY])
    assume(
        config_dict[ConfigKeys.RUNPATH_FILE] != config_dict[ConfigKeys.CONFIG_FILE_KEY]
    )

    return config_dict


def to_config_file(filename, config_dict):
    with open(filename, "w+") as config:
        config.write(
            f"{ConfigKeys.RUNPATH_FILE} {config_dict[ConfigKeys.RUNPATH_FILE]}\n"
        )
        for keyword, keyword_value in config_dict.items():
            if keyword == ConfigKeys.DATA_KW_KEY:
                for define_key, define_value in keyword_value.items():
                    config.write(f"{keyword} {define_key} {define_value}\n")
            elif keyword == ConfigKeys.INSTALL_JOB_DIRECTORY:
                for install_dir in keyword_value:
                    config.write(f"{keyword} {install_dir}\n")
            elif keyword == ConfigKeys.SETENV:
                for statement in keyword_value:
                    config.write(
                        f"{keyword}"
                        f" {statement[ConfigKeys.NAME]} {statement[ConfigKeys.VALUE]}\n"
                    )
            elif keyword == ConfigKeys.ANALYSIS_COPY:
                for statement in keyword_value:
                    config.write(
                        f"{keyword}"
                        f" {statement[ConfigKeys.SRC_NAME]}"
                        f" {statement[ConfigKeys.DST_NAME]}\n"
                    )
            elif keyword == ConfigKeys.ANALYSIS_SET_VAR:
                for statement in keyword_value:
                    config.write(
                        f"{keyword} {statement[ConfigKeys.MODULE_NAME]}"
                        f" {statement[ConfigKeys.VAR_NAME]}"
                        f" {statement[ConfigKeys.VALUE]}\n"
                    )
            elif keyword == ConfigKeys.DEFINE_KEY:
                for define_key, define_value in keyword_value.items():
                    config.write(f"{keyword} {define_key} {define_value}\n")
            elif keyword == ConfigKeys.INSTALL_JOB:
                for job in keyword_value:
                    config.write(
                        f"{keyword}" f" {job[ConfigKeys.NAME]} {job[ConfigKeys.PATH]}\n"
                    )
            elif keyword == ConfigKeys.QUEUE_OPTION:
                for setting in keyword_value:
                    config.write(
                        f"{keyword}"
                        f" {setting[ConfigKeys.NAME]} {setting[ConfigKeys.VALUE]}\n"
                    )
            elif keyword == ConfigKeys.QUEUE_SYSTEM:
                config.write(f"{keyword}" f" {keyword_value.name[:-7]}\n")
            elif keyword == ConfigKeys.UMASK:
                config.write(f"{keyword} 0{keyword_value:o}\n")
            else:
                config.write(f"{keyword} {keyword_value}\n")