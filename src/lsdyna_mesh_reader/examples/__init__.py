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

# https://www.dynaexamples.com/show-cases/joint-screw/
joint_screw = os.path.join(dir_path, "EXP_SC_JOINT_SCREW.key")

# https://www.dynaexamples.com/nvh/example-05-03
wheel = os.path.join(dir_path, "wheel.k")

# https://www.dynaexamples.com/nvh/example-07-01
bracket = os.path.join(dir_path, "bracket.k")

__all__ = [
    "birdball",
    "joint_screw",
    "wheel",
    "bracket",
]
