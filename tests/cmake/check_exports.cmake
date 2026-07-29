if(NOT DEFINED MG_NM OR NOT DEFINED MG_LIBRARY)
  message(FATAL_ERROR "MG_NM and MG_LIBRARY are required")
endif()

execute_process(
  COMMAND "${MG_NM}" -gU "${MG_LIBRARY}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed (${nm_result}): ${nm_error}")
endif()

string(REPLACE "\n" ";" nm_lines "${nm_output}")
set(actual)
foreach(line IN LISTS nm_lines)
  string(STRIP "${line}" line)
  if(NOT line STREQUAL "")
    string(REGEX MATCH "[^ \t]+$" symbol "${line}")
    list(APPEND actual "${symbol}")
  endif()
endforeach()

set(expected
  _mg_bfs
  _mg_graph_create_from_coo
  _mg_graph_destroy
  _mg_graph_num_edges
  _mg_graph_num_vertices
  _mg_khop
  _mg_last_error_message
  _mg_last_run
  _mg_pagerank
  _mg_ppr_topk
  _mg_set_execution
  _mg_version
  _mg_wcc)
list(SORT actual)
list(SORT expected)

if(NOT "${actual}" STREQUAL "${expected}")
  message(FATAL_ERROR
    "Unexpected C ABI exports.\nExpected: ${expected}\nActual: ${actual}")
endif()
