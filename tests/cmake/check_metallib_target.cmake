if(NOT DEFINED MG_STRINGS OR NOT DEFINED MG_METALLIB OR
   NOT DEFINED MG_DEPLOYMENT_TARGET)
  message(FATAL_ERROR
    "MG_STRINGS, MG_METALLIB, and MG_DEPLOYMENT_TARGET are required")
endif()

execute_process(
  COMMAND "${MG_STRINGS}" "${MG_METALLIB}"
  RESULT_VARIABLE strings_result
  OUTPUT_VARIABLE strings_output
  ERROR_VARIABLE strings_error)
if(NOT strings_result EQUAL 0)
  message(FATAL_ERROR
    "strings failed (${strings_result}): ${strings_error}")
endif()

string(REPLACE "." "\\." target_regex "${MG_DEPLOYMENT_TARGET}")
if(NOT strings_output MATCHES "apple-macosx${target_regex}\\.0")
  message(FATAL_ERROR
    "Embedded Metal target does not match macOS ${MG_DEPLOYMENT_TARGET}")
endif()
