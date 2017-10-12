#ifdef CROSSTUSTART
__CROSSTUSTART = DIAG_START_CROSSTU,
#undef CROSSTUSTART
#endif

DIAG(err_fnmap_parsing, CLASS_ERROR, (unsigned)diag::Severity::Error, "error parsing index file: '%0' line: %1 'UniqueID filename' format expected", 0, SFINAE_SubstitutionFailure, false, true, 0)
DIAG(err_multiple_def_index, CLASS_ERROR, (unsigned)diag::Severity::Error, "multiple definitions are found for the same key in index ", 0, SFINAE_SubstitutionFailure, false, true, 0)
