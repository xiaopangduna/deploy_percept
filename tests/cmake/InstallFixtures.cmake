# Install test/demo fixture data alongside tests (share/percept/fixtures/).
if(INSTALL_TESTS)
    install(
        DIRECTORY "${CMAKE_SOURCE_DIR}/tests/fixtures/"
        DESTINATION share/percept/fixtures
        USE_SOURCE_PERMISSIONS)
endif()
