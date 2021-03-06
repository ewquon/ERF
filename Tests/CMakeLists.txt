#=============================================================================
# ERF Testing
#=============================================================================

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

# Set location of gold files according to system/compiler/compiler_version
set(FCOMPARE_GOLD_FILES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ERFGoldFiles/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION})

if(TEST_WITH_FCOMPARE)
  message(STATUS "Test golds directory for fcompare: ${FCOMPARE_GOLD_FILES_DIRECTORY}")
endif()

# Have CMake discover the number of cores on the node
#include(ProcessorCount)
#ProcessorCount(PROCESSES)

#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================

# Standard regression test
function(add_test_r TEST_NAME TEST_EXE_DIR)
    # Set variables for respective binary and source directories for the test
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/erf_${TEST_EXE_DIR})
    # Gold files should be submodule organized by machine and compiler (these are output during configure)
    set(PLOT_GOLD ${FCOMPARE_GOLD_FILES_DIRECTORY}/${TEST_EXE_DIR}/tests/${TEST_NAME}/plt00010)
    # Test plot is currently expected to be after 10 steps
    set(PLOT_TEST ${CURRENT_TEST_BINARY_DIR}/plt00010)
    # Find fcompare
    if(TEST_WITH_FCOMPARE)
      set(FCOMPARE ${CMAKE_BINARY_DIR}/${AMREX_SUBMOD_LOCATION}/Tools/Plotfile/fcompare)
    endif()
    # Make working directory for test
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    # Gather all files in source directory for test
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    # Copy files to test working directory
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    # Set some default runtime options for all tests in this category
    set(RUNTIME_OPTIONS "max_step=10 amr.plot_file=plt amr.checkpoint_files_output=0 amr.plot_files_output=1 amrex.signal_handling=0")
    # Use fcompare to test diffs in plots against gold files
    if(TEST_WITH_FCOMPARE)
      set(FCOMPARE_COMMAND "&& ${FCOMPARE} ${PLOT_GOLD} ${PLOT_TEST}")
    endif()
    if(ERF_ENABLE_MPI)
      set(NP 4)
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
      set(NP 1)
      unset(MPI_COMMANDS)
    endif()
    # Add test and actual test commands to CTest database
    add_test(${TEST_NAME} sh -c "${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i ${RUNTIME_OPTIONS} > ${TEST_NAME}.log ${FCOMPARE_COMMAND}")
    # Set properties for test
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 7200 PROCESSORS ${NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "regression" ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_r)

# Regression tests excluded from CI
function(add_test_re TEST_NAME TEST_EXE_DIR)
    add_test_r(${TEST_NAME} ${TEST_EXE_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "regression;no-ci")
endfunction(add_test_re)

# Verification test with 1 resolution
function(add_test_v1 TEST_NAME TEST_EXE_DIR)
    # Set variables for respective binary and source directories for the test
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/erf_${TEST_EXE_DIR})
    # Make working directory for test
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    # Gather all files in source directory for test
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    # Copy files to test working directory
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    # Get test options
    if(ERF_ENABLE_MPI)
      set(NP 4)
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
      set(NP 1)
      unset(MPI_COMMANDS)
    endif()
    # Set some default runtime options for all tests in this category
    set(RUNTIME_OPTIONS "amr.plot_file=plt amr.checkpoint_files_output=0 amr.plot_files_output=1 amrex.signal_handling=0")
    # Define our main run command
    set(RUN_COMMAND "rm mmslog datlog || true && ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    # Add test and actual test commands to CTest database
    add_test(${TEST_NAME} sh -c "${RUN_COMMAND} ${RUNTIME_OPTIONS} > ${TEST_NAME}.log && nosetests ${TEST_NAME}.py")
    # Set properties for test
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 7200 PROCESSORS ${NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "verification" ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_v1)

# Verification test with multiple resolutions (each test runs on maximum number of processes on node)
function(add_test_v2 TEST_NAME TEST_EXE_DIR LIST_OF_GRID_SIZES)
    # Make sure run command is cleared before we construct it
    unset(MASTER_RUN_COMMAND)
    # Set variables for respective binary and source directories for the test
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/tests/${TEST_NAME})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/erf_${TEST_EXE_DIR})
    # Get last item in resolution list so we can find out when we are on the last item in our loop
    list(GET LIST_OF_GRID_SIZES -1 LAST_GRID_SIZE_IN_LIST)
    # Create the commands to run for each resolution
    foreach(GRID_SIZE IN LISTS LIST_OF_GRID_SIZES)
      # Make working directory for test
      file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE})
      # Gather all files in source directory for test
      file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
      # Copy files to test working directory
      file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}/")
      # Set number of cells at runtime according to dimension
      if(${ERF_DIM} EQUAL 3)
        set(NCELLS "${GRID_SIZE} ${GRID_SIZE} ${GRID_SIZE}")
      elseif(${ERF_DIM} EQUAL 2)
        set(NCELLS "${GRID_SIZE} ${GRID_SIZE}")
      elseif(${ERF_DIM} EQUAL 1)
        set(NCELLS "${GRID_SIZE}")
      endif()
      # Set the command to delete files from previous test runs in each resolution
      set(DELETE_PREVIOUS_FILES_COMMAND "rm mmslog datlog || true")
      if(ERF_ENABLE_MPI)
        set(NP 4)
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
      else()
        set(NP 1)
        unset(MPI_COMMANDS)
      endif()
      # Set the run command for this resolution
      set(RUN_COMMAND_${GRID_SIZE} "${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}/${TEST_NAME}.i")
      # Set some runtime options for each resolution
      set(RUNTIME_OPTIONS_${GRID_SIZE} "amrex.signal_handling=0 amr.plot_file=plt amr.checkpoint_files_output=0 amr.plot_files_output=1 amr.n_cell=${NCELLS}")
      # Construct our large run command with everything &&'d together
      string(APPEND MASTER_RUN_COMMAND "cd ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}")
      string(APPEND MASTER_RUN_COMMAND " && ")
      string(APPEND MASTER_RUN_COMMAND "${DELETE_PREVIOUS_FILES_COMMAND}")
      string(APPEND MASTER_RUN_COMMAND " && ")
      string(APPEND MASTER_RUN_COMMAND "${RUN_COMMAND_${GRID_SIZE}} ${RUNTIME_OPTIONS_${GRID_SIZE}} > ${TEST_NAME}-${GRID_SIZE}.log")
      # Add another " && " unless we are on the last resolution in the list
      if(NOT ${GRID_SIZE} EQUAL ${LAST_GRID_SIZE_IN_LIST})
        string(APPEND MASTER_RUN_COMMAND " && ")
      endif()
    endforeach()
    # Set list of images to be uploaded for verification
    set(IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/p_error.png ${CURRENT_TEST_BINARY_DIR}/rho_error.png ${CURRENT_TEST_BINARY_DIR}/u_error.png)
    if(${ERF_DIM} EQUAL 3)
      list(APPEND IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/v_error.png ${CURRENT_TEST_BINARY_DIR}/w_error.png)
    elseif(${ERF_DIM} EQUAL 2)
      list(APPEND IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/v_error.png)
    endif()
    # Add test and actual test commands to CTest database (need to convert this to arrays for resolutions)
    add_test(${TEST_NAME} sh -c "${MASTER_RUN_COMMAND} && cd ${CURRENT_TEST_BINARY_DIR} && nosetests ${CMAKE_CURRENT_SOURCE_DIR}/test_second_order.py")
    # Set properties for test and make sure test dependencies have run before this test will run
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 28800 PROCESSORS ${NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}" LABELS "verification;no-ci" ATTACHED_FILES "${IMAGES_TO_UPLOAD}")
endfunction(add_test_v2)

# Standard unit test
function(add_test_u TEST_NAME)
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/UnitTests/tests/${TEST_NAME})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/UnitTests/erf_${TEST_NAME})
    # Make working directory for test
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    set(NP 1)
    if(ERF_ENABLE_MPI)
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
      unset(MPI_COMMANDS)
    endif()
    # Add test and commands to CTest database
    add_test(${TEST_NAME} sh -c "${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS}")
    # Set properties for test
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 1800 PROCESSORS ${NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "unit")
endfunction(add_test_u)

#=============================================================================
# Regression tests
#=============================================================================

# Run in CI
if(ERF_DIM GREATER 1)
  add_test_r(tg-1 TG)
  add_test_r(tg-2 TG)
  add_test_r(hit-1 HIT)
  add_test_r(hit-2 HIT)
  add_test_r(hit-3 HIT)
  add_test_r(sod-1 Sod)
endif()
if(ERF_ENABLE_MASA)
  add_test_r(mms-3 MMS)
  if(ERF_DIM GREATER 1)
    add_test_r(mms-1 MMS)
    add_test_r(mms-2 MMS)
    add_test_r(mms-4 MMS)
    if(ERF_DIM GREATER 2)
      add_test_r(mms-5 MMS)
    endif()
    if(ERF_ENABLE_AMREX_EB)
      add_test_r(ebmms-1 EB_MMS)
    endif()
  endif()
endif()

# Not run in CI
if(ERF_DIM GREATER 1)
  add_test_re(pmf-1 PMF)
  add_test_re(multispecsod-1 MultiSpecSod)
  if(ERF_DIM GREATER 2)
    add_test_re(tg-3 TG)
    add_test_re(tg-4 TG)
  endif()
endif()

#=============================================================================
# Verification tests
#=============================================================================
if(ERF_ENABLE_MASA AND (ERF_DIM GREATER 2))
  add_test_v1(symmetry MMS)
  if(ERF_ENABLE_AMREX_EB)
    add_test_v1(eb-symmetry EB_MMS)
  endif()

  # Create list of resolutions we want to test with; make sure to pass it as a string in quotes
  set(LIST_OF_GRID_SIZES 8 12 16 20)
  add_test_v2(cns-no-amr MMS "${LIST_OF_GRID_SIZES}")
  add_test_v2(cns-no-amr-mol MMS "${LIST_OF_GRID_SIZES}")
  #add_test_v3(cns-amr MMS "${LIST_OF_GRID_SIZES}") # This one takes a while with AMR

  set(LIST_OF_GRID_SIZES 8 12 16 24)
  add_test_v2(cns-les-no-amr MMS "${LIST_OF_GRID_SIZES}")
endif()

#=============================================================================
# Unit tests
#=============================================================================
add_test_u(unit_tests)

#=============================================================================
# Performance tests
#=============================================================================
