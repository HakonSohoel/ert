copy_test_files() {
    cp -r "${CI_SOURCE_ROOT}"/tests "${CI_TEST_ROOT}"
    cp -r "${CI_SOURCE_ROOT}"/docs "${CI_TEST_ROOT}"/docs
    ln -s "${CI_SOURCE_ROOT}"/test-data "${CI_TEST_ROOT}"/test-data

    ln -s "${CI_SOURCE_ROOT}"/src "${CI_TEST_ROOT}"/src

    # Trick ERT to find a fake source root
    mkdir "${CI_TEST_ROOT}"/.git

    # Keep pytest configuration:
    ln -s "${CI_SOURCE_ROOT}"/pyproject.toml "${CI_TEST_ROOT}"/pyproject.toml
}

install_test_dependencies() {
    pip install ".[dev]"
}

run_ert_with_opm() {
    pushd "${CI_TEST_ROOT}" || exit 1

    cp -r "${CI_SOURCE_ROOT}/test-data/ert/flow_example" ert_with_opm
    pushd ert_with_opm || exit 1

    ert test_run --disable-monitoring flow.ert ||
        (
            # In case ert fails, print log files if they are there:
            cat spe1_out/realization-0/iter-0/STATUS || true
            cat spe1_out/realization-0/iter-0/ERROR || true
            cat spe1_out/realization-0/iter-0/FLOW.stderr.0 || true
            cat spe1_out/realization-0/iter-0/FLOW.stdout.0 || true
            cat logs/ert-log* || true
        )
    popd || exit 1
}

# Clean up everest egg tmp folders
remove_one_week_old_temp_folders () {
    case "$1" in
        "azure")
            RUNNER_ROOT="/lustre1/users/f_scout_ci/egg_tests"
            ;;
        *)
            RUNNER_ROOT="/scratch/oompf/egg_tests"
            ;;
    esac
    find "$RUNNER_ROOT" -maxdepth 1 -mtime +7 -user f_scout_ci -type d -exec rm -r {} \;
}


make_egg_runpath () {
    case "$1" in
        "azure")
            mkdir -p /lustre1/users/f_scout_ci/egg_tests
            mktemp -d -p /lustre1/users/f_scout_ci/egg_tests
            ;;
        *)
            mkdir -p /scratch/oompf/egg_tests
            mktemp -d -p /scratch/oompf/egg_tests
            ;;
    esac
}


# Run everest egg test on the cluster both onprem and azure
run_everest_egg_test() {

    # Need to copy the egg test to a directory that is accessible by all cluster members
    EGG_RUNPATH=$(make_egg_runpath "$CI_RUNNER_LABEL")
    cp -r "${CI_SOURCE_ROOT}/test-data/everest/egg" "$EGG_RUNPATH"
    chmod -R a+rx "$EGG_RUNPATH"
    pushd "${EGG_RUNPATH}/egg" || exit 1
    echo "EGG_RUNPATH: $EGG_RUNPATH"

    disable_komodo
    # shellcheck source=/dev/null
    source "${_KOMODO_ROOT}/${_FULL_RELEASE_NAME}/enable"

    CONFIG="everest/model/config.yml"
    if [[ "$CI_RUNNER_LABEL" == "azure" ]]; then
        sed -i "s/name: local/name: torque\n    queue: permanent_8/g" "$CONFIG"
        export PATH=$PATH:/opt/pbs/bin
    elif [[ "$CI_RUNNER_LABEL" == "onprem" ]]; then
        sed -i "s/name: local/name: lsf/g" "$CONFIG"
        export PATH=$PATH:/global/bin
    fi

    everest run "$CONFIG" --skip-prompt --debug --disable-monitoring
    STATUS=$?
    popd || exit 1

    remove_one_week_old_temp_folders "$CI_RUNNER_LABEL"

    return $STATUS
}



start_tests() {
    export NO_PROXY=localhost,127.0.0.1
    export ECL_SKIP_SIGNAL=ON
    export ERT_PYTEST_ARGS=--eclipse-simulator

    pushd "${CI_TEST_ROOT}"/tests/ert || exit 1

    if [ "$CI_SUBSYSTEM_TEST" == "ert" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile ert-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "everest" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile everest-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "everest-egg" ]; then
      run_everest_egg_test
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-limit-memory" ]; then
      # Restricting the number of threads utilized by numpy to control memory consumption, as some tests evaluate memory usage and additional threads increase it.
      export OMP_NUM_THREADS=1

      # Run ert tests that evaluates memory consumption
      pytest -n 2 --durations=0 -m "limit_memory" --memray
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-queue-system" ]; then
      basetemp=$(mktemp -d -p "$_ERT_TESTS_SHARED_TMP")
      pytest --timeout=3600 -v --"$_ERT_TESTS_QUEUE_SYSTEM" --basetemp="$basetemp" unit_tests/scheduler
      return_code_ert_scheduler_tests=$?
      rm -rf "$basetemp" || true
      return $return_code_ert_scheduler_tests
    elif [ "$CI_SUBSYSTEM_TEST" == "opm-integration" ]; then
      run_ert_with_opm
      return $?
    fi

    echo "Error: Variable $CI_SUBSYSTEM_TEST did not match any testable subsystem"
    return 1
}
