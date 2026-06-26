#include "deploy_percept/engine/OutputAccess.hpp"

namespace deploy_percept
{
    namespace engine
    {

        OutputAccess::OutputAccess(BaseEngine &engine)
            : engine_(engine),
              views_(engine.borrow_output_views())
        {
        }

        OutputAccess::~OutputAccess()
        {
            engine_.release_output_views();
        }

    } // namespace engine
} // namespace deploy_percept
