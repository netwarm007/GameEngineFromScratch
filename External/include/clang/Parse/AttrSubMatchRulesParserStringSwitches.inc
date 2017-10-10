Optional<attr::SubjectMatchRule> defaultIsAttributeSubjectMatchSubRuleFor(StringRef, bool) {
  return None;
}

Optional<attr::SubjectMatchRule> isAttributeSubjectMatchSubRuleFor_variable(StringRef Name, bool IsUnless) {
  if (IsUnless)
    return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
    Case("is_parameter", attr::SubjectMatchRule_variable_not_is_parameter).
    Default(None);
  return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
  Case("is_thread_local", attr::SubjectMatchRule_variable_is_thread_local).
  Case("is_global", attr::SubjectMatchRule_variable_is_global).
  Case("is_parameter", attr::SubjectMatchRule_variable_is_parameter).
  Default(None);
}

Optional<attr::SubjectMatchRule> isAttributeSubjectMatchSubRuleFor_function(StringRef Name, bool IsUnless) {
  if (IsUnless)
    return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
    Default(None);
  return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
  Case("is_member", attr::SubjectMatchRule_function_is_member).
  Default(None);
}

Optional<attr::SubjectMatchRule> isAttributeSubjectMatchSubRuleFor_hasType(StringRef Name, bool IsUnless) {
  if (IsUnless)
    return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
    Default(None);
  return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
  Case("functionType", attr::SubjectMatchRule_hasType_functionType).
  Default(None);
}

Optional<attr::SubjectMatchRule> isAttributeSubjectMatchSubRuleFor_record(StringRef Name, bool IsUnless) {
  if (IsUnless)
    return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
    Case("is_union", attr::SubjectMatchRule_record_not_is_union).
    Default(None);
  return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
  Default(None);
}

Optional<attr::SubjectMatchRule> isAttributeSubjectMatchSubRuleFor_objc_method(StringRef Name, bool IsUnless) {
  if (IsUnless)
    return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
    Default(None);
  return llvm::StringSwitch<Optional<attr::SubjectMatchRule>>(Name).
  Case("is_instance", attr::SubjectMatchRule_objc_method_is_instance).
  Default(None);
}

std::pair<Optional<attr::SubjectMatchRule>, Optional<attr::SubjectMatchRule> (*)(StringRef, bool)> isAttributeSubjectMatchRule(StringRef Name) {
  return llvm::StringSwitch<std::pair<Optional<attr::SubjectMatchRule>, Optional<attr::SubjectMatchRule> (*) (StringRef, bool)>>(Name).
  Case("block", std::make_pair(attr::SubjectMatchRule_block, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("enum", std::make_pair(attr::SubjectMatchRule_enum, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("enum_constant", std::make_pair(attr::SubjectMatchRule_enum_constant, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("field", std::make_pair(attr::SubjectMatchRule_field, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("function", std::make_pair(attr::SubjectMatchRule_function, isAttributeSubjectMatchSubRuleFor_function)).
  Case("namespace", std::make_pair(attr::SubjectMatchRule_namespace, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("objc_category", std::make_pair(attr::SubjectMatchRule_objc_category, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("objc_interface", std::make_pair(attr::SubjectMatchRule_objc_interface, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("objc_method", std::make_pair(attr::SubjectMatchRule_objc_method, isAttributeSubjectMatchSubRuleFor_objc_method)).
  Case("objc_property", std::make_pair(attr::SubjectMatchRule_objc_property, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("objc_protocol", std::make_pair(attr::SubjectMatchRule_objc_protocol, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("record", std::make_pair(attr::SubjectMatchRule_record, isAttributeSubjectMatchSubRuleFor_record)).
  Case("hasType", std::make_pair(attr::SubjectMatchRule_hasType_abstract, isAttributeSubjectMatchSubRuleFor_hasType)).
  Case("type_alias", std::make_pair(attr::SubjectMatchRule_type_alias, defaultIsAttributeSubjectMatchSubRuleFor)).
  Case("variable", std::make_pair(attr::SubjectMatchRule_variable, isAttributeSubjectMatchSubRuleFor_variable)).
  Default(std::make_pair(None, defaultIsAttributeSubjectMatchSubRuleFor));
}

const char *validAttributeSubjectMatchSubRules(attr::SubjectMatchRule Rule) {
  switch (Rule) {
  case attr::SubjectMatchRule_variable:
  return "'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'";
  case attr::SubjectMatchRule_function:
  return "'is_member'";
  case attr::SubjectMatchRule_hasType_abstract:
  return "'functionType'";
  case attr::SubjectMatchRule_record:
  return "'unless(is_union)'";
  case attr::SubjectMatchRule_objc_method:
  return "'is_instance'";
  default: return nullptr;
  }
}

