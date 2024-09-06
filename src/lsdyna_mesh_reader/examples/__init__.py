"""LS-DYNA examples module.

Content reused based on the following implicit open to use license from
https://www.dynaexamples.com/

    The input files and several class notes are available for download. The
    download is free of charge, a login is not required. All examples are
    presented with a brief description. You may find an example by checking a
    specific class or by using the search functionality of the site.

    The content is prepared for educational purposes. Hence, material
    properties and other parameters might be non-physic for simplification.

"""

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# https://www.dynaexamples.com/introduction/intro-by-a.-tabiei/contact/contact-eroding-i
birdball = os.path.join(dir_path, "birdball.k")

__all__ = [
    "birdball",
]
