#include "app/App.h"

#include <exception>
#include <iostream>

int main()
{
    try {
        vollenia::App app;
        return app.run();
    } catch (const std::exception& exception) {
        std::cerr << "VolLenia Playground failed: " << exception.what() << '\n';
        return 1;
    }
}
