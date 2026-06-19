# Wrap a text file into a C header as a raw-string char[]. Usage:
#   cmake -DIN=<src> -DOUT=<hdr> -DVAR=<symbol> -P cmake/embed_text.cmake
file(READ "${IN}" _content)
# Guard against the raw-string delimiter appearing in the source.
if(_content MATCHES "\\)AJIHIP\"")
    message(FATAL_ERROR "embed_text: source contains the )AJIHIP\" delimiter")
endif()
file(WRITE "${OUT}" "// Generated from ${IN}. Do not edit.\n")
file(APPEND "${OUT}" "static const char ${VAR}[] = R\"AJIHIP(\n")
file(APPEND "${OUT}" "${_content}")
file(APPEND "${OUT}" ")AJIHIP\";\n")
