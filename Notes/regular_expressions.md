# Regular Expressions
- One of the unsung successes in __standardization__ in computer science has been __regular expressions (RE)__, language for specifying text search strings.
- Formally, a regular expression is an __algebric notation__ for characterizing a set of strings. 
- Particularly useful for searching in texts, when we have a __pattern__ to search for and a __corpus__ of texts to search throgh.
- A regular expression _search function_ will search throgh the corpus, returning all texts that match the given pattern.
- The simplest kind of regular expression is a sequence of simple characters. 
- To search for woodchuck, we type /woodchuck/. The expression /Buttercup/ matches any string containing the substring Buttercup.
- Regular expressions are case sensitive; lower case /s/ is distinct from upper case /S/ (/s/ matches a lower case s but not an upper case S). 
- This means that the pattern /woodchucks/ will not match the string Woodchucks. We can solve this problem with the use of the square braces [ and ]. The string of characters inside the braces specifies a disjunction of characters to match. For example. pattern __/[wW]/__ matches patterns containing either w or W.
<br><br>
- In cases where there is a well-defined sequence associated with a set of characters, the brackets can be used with the dash (-) to specify any one character in a range.
- The pattern /[2-5]/ specifies any one of the characters 2, 3, 4, or 5. The pattern /[b-g]/ specifies one of the characters b, c, d, e, f, or g.
