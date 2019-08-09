.. _builtin_entities:

Supported builtin entities
==========================

:ref:`Builtin entities <builtin_entity_resolution>` are entities that have
a built-in support in Snips NLU. These entities are associated to specific
builtin entity parsers which provide an extra resolution step. Typically,
dates written in natural language (``"in three days"``) are resolved into ISO
formatted dates (``"2019-08-12 00:00:00 +02:00"``).

Here is the list of supported builtin entities:


+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| Entity                                           | Identifier                                       | Category                                         | Supported Languages                              |
+==================================================+==================================================+==================================================+==================================================+
| `AmountOfMoney`_                                 | snips/amountOfMoney                              | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Duration`_                                      | snips/duration                                   | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Number`_                                        | snips/number                                     | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Ordinal`_                                       | snips/ordinal                                    | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Temperature`_                                   | snips/temperature                                | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Datetime`_                                      | snips/datetime                                   | `Grammar Entity`_                                | de, en, es, fr, it, ja, ko, pt_br, pt_pt         |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Date`_                                          | snips/date                                       | `Grammar Entity`_                                | en                                               |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Time`_                                          | snips/time                                       | `Grammar Entity`_                                | en                                               |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `DatePeriod`_                                    | snips/datePeriod                                 | `Grammar Entity`_                                | en                                               |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `TimePeriod`_                                    | snips/timePeriod                                 | `Grammar Entity`_                                | en                                               |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Percentage`_                                    | snips/percentage                                 | `Grammar Entity`_                                | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `MusicAlbum`_                                    | snips/musicAlbum                                 | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `MusicArtist`_                                   | snips/musicArtist                                | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `MusicTrack`_                                    | snips/musicTrack                                 | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `City`_                                          | snips/city                                       | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Country`_                                       | snips/country                                    | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
| `Region`_                                        | snips/region                                     | `Gazetteer Entity`_                              | de, en, es, fr, it, ja, pt_br, pt_pt             |
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+

The entity identifier (second column above) is what is used in the dataset to
reference a builtin entity.

Grammar Entity
--------------

Grammar entities, in the context of Snips NLU, correspond to entities which 
contain significant `compositionality`_. The semantic meaning of such an 
entity is determined by the meanings of its constituent expressions and the 
rules used to combine them. Modern semantic parsers for these entities are 
often based on defining a formal grammar. In the case of Snips NLU, the parser 
used to handle these entities is `Rustling`_, a Rust adaptation of Facebook's 
`duckling`_.

Gazetteer Entity
----------------

Gazetteer entities correspond to all the builtin entities which do not contain 
any semantic structure, as opposed to the grammar entities. For such 
entities, a `gazetteer entity parser`_ is used to perform the parsing.

Results Examples
----------------

The following sections provide examples for each builtin entity. 

-------------
AmountOfMoney
-------------

Input examples:

.. code-block:: json

   [
     "$10",
     "six euros",
     "around 5€",
     "ten dollars and five cents"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "AmountOfMoney",
       "value": 10.05,
       "precision": "Approximate",
       "unit": "€"
     }
   ]

--------
Duration
--------

Input examples:

.. code-block:: json

   [
     "1h",
     "during two minutes",
     "for 20 seconds",
     "3 months",
     "half an hour",
     "8 years and two days"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Duration",
       "years": 0,
       "quarters": 0,
       "months": 3,
       "weeks": 0,
       "days": 0,
       "hours": 0,
       "minutes": 0,
       "seconds": 0,
       "precision": "Exact"
     }
   ]

------
Number
------

Input examples:

.. code-block:: json

   [
     "2001",
     "twenty one",
     "three hundred and four"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Number",
       "value": 42.0
     }
   ]

-------
Ordinal
-------

Input examples:

.. code-block:: json

   [
     "1st",
     "the second",
     "the twenty third"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Ordinal",
       "value": 2
     }
   ]

-----------
Temperature
-----------

Input examples:

.. code-block:: json

   [
     "70K",
     "3°C",
     "Twenty three degrees",
     "one hundred degrees fahrenheit"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Temperature",
       "value": 23.0,
       "unit": "celsius"
     },
     {
       "kind": "Temperature",
       "value": 60.0,
       "unit": "fahrenheit"
     }
   ]

--------
Datetime
--------

Input examples:

.. code-block:: json

   [
     "Today",
     "at 8 a.m.",
     "4:30 pm",
     "in 1 hour",
     "the 3rd tuesday of June"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "InstantTime",
       "value": "2017-06-13 18:00:00 +02:00",
       "grain": "Hour",
       "precision": "Exact"
     },
     {
       "kind": "TimeInterval",
       "from": "2017-06-07 18:00:00 +02:00",
       "to": "2017-06-08 00:00:00 +02:00"
     }
   ]

----
Date
----

Input examples:

.. code-block:: json

   [
     "today",
     "on Wednesday",
     "March 26th",
     "saturday january 19",
     "monday 15th april 2019",
     "the day after tomorrow"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "InstantTime",
       "value": "2017-06-13 00:00:00 +02:00",
       "grain": "Day",
       "precision": "Exact"
     }
   ]

----
Time
----

Input examples:

.. code-block:: json

   [
     "now",
     "at noon",
     "at 8 a.m.",
     "4:30 pm",
     "in one hour",
     "for ten o'clock",
     "at ten in the evening"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "InstantTime",
       "value": "2017-06-13 18:00:00 +02:00",
       "grain": "Hour",
       "precision": "Exact"
     }
   ]

----------
DatePeriod
----------

Input examples:

.. code-block:: json

   [
     "january",
     "2019",
     "from monday to friday",
     "from wednesday 27th to saturday 30th",
     "this week"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "TimeInterval",
       "from": "2017-06-07 00:00:00 +02:00",
       "to": "2017-06-09 00:00:00 +02:00"
     }
   ]

----------
TimePeriod
----------

Input examples:

.. code-block:: json

   [
     "until dinner",
     "from five to ten",
     "by the end of the day"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "TimeInterval",
       "from": "2017-06-07 18:00:00 +02:00",
       "to": "2017-06-07 20:00:00 +02:00"
     }
   ]

----------
Percentage
----------

Input examples:

.. code-block:: json

   [
     "25%",
     "twenty percent",
     "two hundred and fifty percents"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Percentage",
       "value": 20.0
     }
   ]

----------
MusicAlbum
----------

Input examples:

.. code-block:: json

   [
     "Discovery"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "MusicAlbum",
       "value": "Discovery"
     }
   ]

-----------
MusicArtist
-----------

Input examples:

.. code-block:: json

   [
     "Daft Punk"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "MusicArtist",
       "value": "Daft Punk"
     }
   ]

----------
MusicTrack
----------

Input examples:

.. code-block:: json

   [
     "Harder Better Faster Stronger"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "MusicTrack",
       "value": "Harder Better Faster Stronger"
     }
   ]

----
City
----

Input examples:

.. code-block:: json

   [
     "San Francisco",
     "Los Angeles",
     "Beijing",
     "Paris"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "City",
       "value": "Paris"
     }
   ]

-------
Country
-------

Input examples:

.. code-block:: json

   [
     "France"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Country",
       "value": "France"
     }
   ]

------
Region
------

Input examples:

.. code-block:: json

   [
     "California",
     "Washington"
   ]


Output examples:

.. code-block:: json

   [
     {
       "kind": "Region",
       "value": "California"
     }
   ]


.. _compositionality: https://en.wikipedia.org/wiki/Principle_of_compositionality
.. _Rustling: https://github.com/snipsco/rustling-ontology
.. _duckling: https://github.com/facebook/duckling
.. _gazetteer entity parser: https://github.com/snipsco/gazetteer-entity-parser
