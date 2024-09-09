"""LS-DYNA Mesh Reader examples module.

Content reused based on the following implicit open to use license from
`LS-DYNA Examples <https://www.dynaexamples.com/>`_.

**Usage Notice from LS-DYNA**

    The input files and several class notes are available for download. The
    download is free of charge, a login is not required. All examples are
    presented with a brief description. You may find an example by checking a
    specific class or by using the search functionality of the site.

    The content is prepared for educational purposes. Hence, material
    properties and other parameters might be non-physic for simplification.

    Copyright (c) 2014 DYNAmore GmbH (www.dynamore.de)
    Copying for non-commercial usage allowed, if this notice is included.


Example Files
^^^^^^^^^^^^^

.. py:data:: lsdyna_mesh_reader.examples.birdball
   :annotation: = str

   Path to the `Contact Eroding I <https://www.dynaexamples.com/introduction/intro-by-a.-tabiei/contact/contact-eroding-i>`_ example file. Usage:

   .. code:: pycon

      >>> from lsdyna_mesh_reader import examples
      >>> import lsdyna_mesh_reader
      >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
      >>> deck
      LSDYNA Deck with:
        Node sections:              1
        Element Solid sections:     1
        Element Shell sections:     1

.. py:data:: lsdyna_mesh_reader.examples.joint_screw
   :annotation: = str

   Path to the `Joint Screw <https://www.dynaexamples.com/show-cases/joint-screw/>`_ example file. Usage:

   .. code:: pycon

      >>> from lsdyna_mesh_reader import examples
      >>> import lsdyna_mesh_reader
      >>> deck = lsdyna_mesh_reader.Deck(examples.joint_screw)
      >>> deck
      LSDYNA Deck with:
        Node sections:              1
        Element Solid sections:     1
        Element Shell sections:     1


.. py:data:: lsdyna_mesh_reader.examples.wheel
   :annotation: = str

   Path to the `SSD with a wheel rim <https://www.dynaexamples.com/nvh/example-05-03>`_ example file. Usage:

   .. code:: pycon

      >>> from lsdyna_mesh_reader import examples
      >>> import lsdyna_mesh_reader
      >>> deck = lsdyna_mesh_reader.Deck(examples.wheel)
      >>> deck
      LSDYNA Deck with:
        Node sections:              1
        Element Solid sections:     0
        Element Shell sections:     1


.. py:data:: lsdyna_mesh_reader.examples.bracket
   :annotation: = str

   Path to `An aluminium bracket <https://www.dynaexamples.com/nvh/example-05-03>`_ example file. Usage:

   .. code:: pycon

      >>> from lsdyna_mesh_reader import examples
      >>> import lsdyna_mesh_reader
      >>> deck = lsdyna_mesh_reader.Deck(examples.bracket)
      >>> deck
      LSDYNA Deck with:
        Node sections:              1
        Element Solid sections:     0
        Element Shell sections:     1


.. py:data:: lsdyna_mesh_reader.examples.simple_plate
   :annotation: = str

   Path to the `Square Plate Out-of-Plane Vibration (thick shell mesh) <https://www.dynaexamples.com/nvh/example-05-03>`_ example file. Usage:

   .. code:: pycon

      >>> from lsdyna_mesh_reader import examples
      >>> import lsdyna_mesh_reader
      >>> deck = lsdyna_mesh_reader.Deck(examples.simple_plate)
      >>> deck
      LSDYNA Deck with:
        Node sections:              1
        Element Solid sections:     0
        Element Shell sections:     1

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

# https://www.dynaexamples.com/introduction/Introduction/example-13
simple_plate = os.path.join(dir_path, "ex_13_thick_shell_elform_2.k")

__all__ = [
    "birdball",
    "joint_screw",
    "wheel",
    "bracket",
    "simple_plate",
]
