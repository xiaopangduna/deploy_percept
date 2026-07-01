include(GNUInstallDirs)

install(TARGETS deploy_percept_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TARGET deploy_percept_utils)
    install(TARGETS deploy_percept_utils
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(DIRECTORY include/deploy_percept
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})