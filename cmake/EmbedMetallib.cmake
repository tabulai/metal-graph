# Embed a binary file as a C array: cmake -DIN=... -DOUT=... -P EmbedMetallib.cmake
file(READ ${IN} _hex HEX)
string(LENGTH ${_hex} _hexlen)
math(EXPR _bytes "${_hexlen} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _array ${_hex})
file(WRITE ${OUT}
"/* Auto-generated: embedded mg_default.metallib (${_bytes} bytes). */
#include <stddef.h>
const unsigned char mg_metallib_data[] = {${_array}};
const size_t mg_metallib_size = sizeof(mg_metallib_data);
")
