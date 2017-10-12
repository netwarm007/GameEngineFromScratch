/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Parsed attribute helpers                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static bool defaultAppertainsTo(Sema &, const AttributeList &,const Decl *) {
  return true;
}

static bool defaultDiagnoseLangOpts(Sema &, const AttributeList &) {
  return true;
}

static bool defaultTargetRequirements(const TargetInfo &) {
  return true;
}

static unsigned defaultSpellingIndexToSemanticSpelling(const AttributeList &Attr) {
  return UINT_MAX;
}

static bool checkAMDGPUFlatWorkGroupSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedKernelFunction;
    return false;
  }
  return true;
}

static void matchRulesForAMDGPUFlatWorkGroupSize(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkAMDGPUNumSGPRAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedKernelFunction;
    return false;
  }
  return true;
}

static void matchRulesForAMDGPUNumSGPR(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkAMDGPUNumVGPRAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedKernelFunction;
    return false;
  }
  return true;
}

static void matchRulesForAMDGPUNumVGPR(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkAMDGPUWavesPerEUAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedKernelFunction;
    return false;
  }
  return true;
}

static void matchRulesForAMDGPUWavesPerEU(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool isTargetarmthumbarmebthumbebavrx86x86_64msp430mipsmipsel(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::armeb || T.getArch() == llvm::Triple::thumbeb || T.getArch() == llvm::Triple::avr || T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::msp430 || T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

static bool checkAVRSignalAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool isTargetavr(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::avr);
}

static void matchRulesForAVRSignal(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool isStruct(const Decl *D) {
  if (const auto *S = dyn_cast<RecordDecl>(D))
    return !S->isUnion();
  return false;
}

static bool checkAbiTagAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isStruct(D) && !isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<NamespaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedStructClassVariableFunctionOrInlineNamespace;
    return false;
  }
  return true;
}

static void matchRulesForAbiTag(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record_not_is_union, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
}

static bool checkAcquireCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned AcquireCapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_acquire_capability = 0,
    CXX11_clang_acquire_capability = 1,
    GNU_acquire_shared_capability = 2,
    CXX11_clang_acquire_shared_capability = 3,
    GNU_exclusive_lock_function = 4,
    GNU_shared_lock_function = 5
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_acquire_capability;
    case 1: return CXX11_clang_acquire_capability;
    case 2: return GNU_acquire_shared_capability;
    case 3: return CXX11_clang_acquire_shared_capability;
    case 4: return GNU_exclusive_lock_function;
    case 5: return GNU_shared_lock_function;
  }
}

static bool isSharedVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage() && !S->getTLSKind();
  return false;
}

static bool checkAcquiredAfterAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkAcquiredBeforeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool isGlobalVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->hasGlobalStorage();
  return false;
}

static bool checkAliasAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkAlignValueAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrTypedef;
    return false;
  }
  return true;
}

static void matchRulesForAlignValue(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
}

static unsigned AlignedAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_aligned = 0,
    CXX11_gnu_aligned = 1,
    Declspec_align = 2,
    Keyword_alignas = 3,
    Keyword_Alignas = 4
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_aligned;
    case 1: return CXX11_gnu_aligned;
    case 2: return Declspec_align;
    case 3: return Keyword_alignas;
    case 4: return Keyword_Alignas;
  }
}

static bool isHasFunctionProto(const Decl *D) {
  if (const auto *S = dyn_cast<Decl>(D))
    return (S->getFunctionType(true) != nullptr &&
                              isa<FunctionProtoType>(S->getFunctionType())) ||
                                       isa<ObjCMethodDecl>(S) ||
                                       isa<BlockDecl>(S);
  return false;
}

static bool checkAllocAlignAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionWithProtoType;
    return false;
  }
  return true;
}

static bool checkAllocSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForAllocSize(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkAlwaysInlineAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned AlwaysInlineAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_always_inline = 0,
    CXX11_gnu_always_inline = 1,
    Keyword_forceinline = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_always_inline;
    case 1: return CXX11_gnu_always_inline;
    case 2: return Keyword_forceinline;
  }
}

static void matchRulesForAnnotate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
}

static bool isTargetx86x86_64(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64);
}

static bool checkArcWeakrefUnavailableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static unsigned ArgumentWithTypeTagAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_argument_with_type_tag = 0,
    GNU_pointer_with_type_tag = 1
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_argument_with_type_tag;
    case 1: return GNU_pointer_with_type_tag;
  }
}

static bool checkAssertCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned AssertCapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_assert_capability = 0,
    CXX11_clang_assert_capability = 1,
    GNU_assert_shared_capability = 2,
    CXX11_clang_assert_shared_capability = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_assert_capability;
    case 1: return CXX11_clang_assert_capability;
    case 2: return GNU_assert_shared_capability;
    case 3: return CXX11_clang_assert_shared_capability;
  }
}

static bool checkAssertExclusiveLockAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkAssertSharedLockAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkAssumeAlignedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static void matchRulesForAssumeAligned(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkAvailabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<NamedDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedNamedDecl;
    return false;
  }
  return true;
}

static void matchRulesForAvailability(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum_constant, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_category, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

static bool checkCFAuditedTransferAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkCFConsumedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static bool checkCFUnknownTransferAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkCUDAConstantAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkCUDALangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.CUDA)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static bool checkCUDADeviceAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return false;
  }
  return true;
}

static bool checkCUDAGlobalAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkCUDAHostAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkCUDAInvalidTargetAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool isFunctionLike(const Decl *D) {
  if (const auto *S = dyn_cast<Decl>(D))
    return S->getFunctionType(false) != nullptr;
  return false;
}

static bool checkCUDALaunchBoundsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static bool checkCUDASharedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkCXX11NoReturnAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForCXX11NoReturn(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkCallableWhenAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static void matchRulesForCallableWhen(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

static bool checkCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedStructOrUnionOrTypedef;
    return false;
  }
  return true;
}

static unsigned CapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_capability = 0,
    CXX11_clang_capability = 1,
    GNU_shared_capability = 2,
    CXX11_clang_shared_capability = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_capability;
    case 1: return CXX11_clang_capability;
    case 2: return GNU_shared_capability;
    case 3: return CXX11_clang_shared_capability;
  }
}

static bool checkCarriesDependencyAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrParameter;
    return false;
  }
  return true;
}

static void matchRulesForCarriesDependency(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkCleanupAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkColdAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkCommonAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkConstructorAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkConsumableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static void matchRulesForConsumable(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

static bool checkConsumableAutoCastAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool checkConsumableSetOnReadAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool checkConvergentAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForConvergent(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkDLLExportAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D) && !isa<CXXRecordDecl>(D) && !isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus     ? ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)            ? ExpectedFunctionVariableClassOrObjCInterface            : ExpectedFunctionVariableOrClass)     : ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)            ? ExpectedFunctionVariableOrObjCInterface            : ExpectedVariableOrFunction));
    return false;
  }
  return true;
}

static bool isTargetx86x86_64armthumbaarch64Win32(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && (T.getOS() == llvm::Triple::Win32);
}

static void matchRulesForDLLExport(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkDLLImportAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D) && !isa<CXXRecordDecl>(D) && !isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus     ? ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)            ? ExpectedFunctionVariableClassOrObjCInterface            : ExpectedFunctionVariableOrClass)     : ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)            ? ExpectedFunctionVariableOrObjCInterface            : ExpectedVariableOrFunction));
    return false;
  }
  return true;
}

static void matchRulesForDLLImport(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkDestructorAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkDiagnoseIfAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethodOrProperty;
    return false;
  }
  return true;
}

static bool checkDisableTailCallsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static void matchRulesForDisableTailCalls(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkEmptyBasesAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool isTargetx86x86_64armthumbaarch64Microsoft(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::x86 || T.getArch() == llvm::Triple::x86_64 || T.getArch() == llvm::Triple::arm || T.getArch() == llvm::Triple::thumb || T.getArch() == llvm::Triple::aarch64) && (Target.getCXXABI().getKind() == TargetCXXABI::Microsoft);
}

static bool checkEnableIfAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForEnableIf(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkEnumExtensibilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedEnum;
    return false;
  }
  return true;
}

static void matchRulesForEnumExtensibility(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
}

static bool checkExclusiveTrylockFunctionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkExtVectorTypeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedType;
    return false;
  }
  return true;
}

static bool checkExternalSourceSymbolAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<NamedDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedNamedDecl;
    return false;
  }
  return true;
}

static void matchRulesForExternalSourceSymbol(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum_constant, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_field, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_namespace, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_category, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_type_alias, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

static bool checkFlagEnumAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedEnum;
    return false;
  }
  return true;
}

static void matchRulesForFlagEnum(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
}

static bool checkFlattenAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForFlatten(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkFormatAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isa<BlockDecl>(D) && !isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionWithProtoType;
    return false;
  }
  return true;
}

static bool checkFormatArgAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionWithProtoType;
    return false;
  }
  return true;
}

static bool checkGNUInlineAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkGuardedByAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkGuardedVarAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkHotAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool isObjCInstanceMethod(const Decl *D) {
  if (const auto *S = dyn_cast<ObjCMethodDecl>(D))
    return S->isInstanceMethod();
  return false;
}

static bool checkIBActionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isObjCInstanceMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjCInstanceMethod;
    return false;
  }
  return true;
}

static bool checkIFuncAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForIFunc(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkInitPriorityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkInternalLinkageAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass : ExpectedVariableOrFunction);
    return false;
  }
  return true;
}

static void matchRulesForInternalLinkage(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

static bool checkLTOVisibilityPublicAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static void matchRulesForLTOVisibilityPublic(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

static bool checkLayoutVersionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool checkLockReturnedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkLockableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool checkLocksExcludedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned LoopHintAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    Pragma_clang_loop = 0,
    Pragma_unroll = 1,
    Pragma_nounroll = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Pragma_clang_loop;
    case 1: return Pragma_unroll;
    case 2: return Pragma_nounroll;
  }
}

static bool checkMicrosoftExtLangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.MicrosoftExt)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static unsigned MSInheritanceAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    Keyword_single_inheritance = 0,
    Keyword_multiple_inheritance = 1,
    Keyword_virtual_inheritance = 2,
    Keyword_unspecified_inheritance = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_single_inheritance;
    case 1: return Keyword_multiple_inheritance;
    case 2: return Keyword_virtual_inheritance;
    case 3: return Keyword_unspecified_inheritance;
  }
}

static bool checkMSNoVTableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool checkMSStructAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool checkMicroMipsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool isTargetmipsmipsel(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel);
}

static void matchRulesForMicroMips(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkMinSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static bool checkMips16AppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkMipsLongCallAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool isTargetmipsmipselmips64mips64el(const TargetInfo &Target) {
  const llvm::Triple &T = Target.getTriple();
  return (T.getArch() == llvm::Triple::mips || T.getArch() == llvm::Triple::mipsel || T.getArch() == llvm::Triple::mips64 || T.getArch() == llvm::Triple::mips64el);
}

static unsigned MipsLongCallAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_long_call = 0,
    CXX11_gnu_long_call = 1,
    GNU_far = 2,
    CXX11_gnu_far = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_long_call;
    case 1: return CXX11_gnu_long_call;
    case 2: return GNU_far;
    case 3: return CXX11_gnu_far;
  }
}

static void matchRulesForMipsLongCall(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkMipsShortCallAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned MipsShortCallAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_short_call = 0,
    CXX11_gnu_short_call = 1,
    GNU_near = 2,
    CXX11_gnu_near = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_short_call;
    case 1: return CXX11_gnu_short_call;
    case 2: return GNU_near;
    case 3: return CXX11_gnu_near;
  }
}

static void matchRulesForMipsShortCall(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkModeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<EnumDecl>(D) && !isa<TypedefNameDecl>(D) && !isa<FieldDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableEnumFieldOrTypedef;
    return false;
  }
  return true;
}

static bool checkNSConsumedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static bool checkNSConsumesSelfAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static bool checkNakedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNoAliasAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNoCommonAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool isNonParmVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->getKind() != Decl::ParmVar;
  return false;
}

static bool checkNoDebugAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isFunctionLike(D) && !isa<ObjCMethodDecl>(D) && !isNonParmVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return false;
  }
  return true;
}

static void matchRulesForNoDebug(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_not_is_parameter, /*IsSupported=*/true));
}

static bool checkNoDuplicateAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForNoDuplicate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkNoInlineAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNoInstrumentFunctionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNoMicroMipsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForNoMicroMips(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkNoMips16AppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNoSanitizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrGlobalVar;
    return false;
  }
  return true;
}

static void matchRulesForNoSanitize(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

static bool checkNoSanitizeSpecificAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrGlobalVar;
    return false;
  }
  return true;
}

static void matchRulesForNoSanitizeSpecific(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

static bool checkNoSplitStackAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForNoSplitStack(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkNoThreadSafetyAnalysisAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkNonNullAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isHasFunctionProto(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrParameter;
    return false;
  }
  return true;
}

static bool checkNotTailCalledAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForNotTailCalled(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkObjCBoxableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedStructOrUnion;
    return false;
  }
  return true;
}

static void matchRulesForObjCBoxable(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
}

static bool checkObjCBridgeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedStructOrUnionOrTypedef;
    return false;
  }
  return true;
}

static bool checkObjCBridgeMutableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool checkObjCBridgeRelatedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool isObjCInterfaceDeclInitMethod(const Decl *D) {
  if (const auto *S = dyn_cast<ObjCMethodDecl>(D))
    return S->getMethodFamily() == OMF_init &&
                                 (isa<ObjCInterfaceDecl>(S->getDeclContext()) ||
                                  (isa<ObjCCategoryDecl>(S->getDeclContext()) &&
            cast<ObjCCategoryDecl>(S->getDeclContext())->IsClassExtension()));
  return false;
}

static bool checkObjCDesignatedInitializerAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isObjCInterfaceDeclInitMethod(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjCInterfaceDeclInitMethod;
    return false;
  }
  return true;
}

static bool checkObjCExceptionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static bool checkObjCExplicitProtocolImplAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCProtocol;
    return false;
  }
  return true;
}

static bool checkObjCMethodFamilyAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static void matchRulesForObjCMethodFamily(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkObjCPreciseLifetimeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkObjCRequiresPropertyDefsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static bool checkObjCRequiresSuperAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static void matchRulesForObjCRequiresSuper(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkObjCReturnsInnerPointerAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethodOrProperty;
    return false;
  }
  return true;
}

static bool checkObjCRootClassAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static bool checkObjCRuntimeNameAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D) && !isa<ObjCProtocolDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterfaceOrProtocol;
    return false;
  }
  return true;
}

static void matchRulesForObjCRuntimeName(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_protocol, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkObjCRuntimeVisibleAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static void matchRulesForObjCRuntimeVisible(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkObjCSubclassingRestrictedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedObjectiveCInterface;
    return false;
  }
  return true;
}

static void matchRulesForObjCSubclassingRestricted(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_interface, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkOpenCLAccessAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D) && !isa<TypedefNameDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameterOrTypedef;
    return false;
  }
  return true;
}

static unsigned OpenCLAccessAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    Keyword_read_only = 0,
    Keyword_write_only = 2,
    Keyword_read_write = 4
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Keyword_read_only;
    case 1: return Keyword_read_only;
    case 2: return Keyword_write_only;
    case 3: return Keyword_write_only;
    case 4: return Keyword_read_write;
    case 5: return Keyword_read_write;
  }
}

static bool checkOpenCLIntelReqdSubGroupSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForOpenCLIntelReqdSubGroupSize(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkOpenCLKernelAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkOpenCLNoSVMAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkOpenCLLangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.OpenCL)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static void matchRulesForOpenCLNoSVM(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable, /*IsSupported=*/true));
}

static bool checkOptimizeNoneAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static void matchRulesForOptimizeNone(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkOverloadableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForOverloadable(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkOwnershipAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isHasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionWithProtoType;
    return false;
  }
  return true;
}

static unsigned OwnershipAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_ownership_holds = 0,
    GNU_ownership_returns = 1,
    GNU_ownership_takes = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_ownership_holds;
    case 1: return GNU_ownership_returns;
    case 2: return GNU_ownership_takes;
  }
}

static bool checkParamTypestateAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static void matchRulesForParamTypestate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool checkPassObjectSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static void matchRulesForPassObjectSize(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool checkPragmaClangBSSSectionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkPragmaClangDataSectionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkPragmaClangRodataSectionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkPragmaClangTextSectionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkPtGuardedByAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkPtGuardedVarAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FieldDecl>(D) && !isSharedVar(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return false;
  }
  return true;
}

static bool checkReleaseCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned ReleaseCapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_release_capability = 0,
    CXX11_clang_release_capability = 1,
    GNU_release_shared_capability = 2,
    CXX11_clang_release_shared_capability = 3,
    GNU_release_generic_capability = 4,
    CXX11_clang_release_generic_capability = 5,
    GNU_unlock_function = 6
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_release_capability;
    case 1: return CXX11_clang_release_capability;
    case 2: return GNU_release_shared_capability;
    case 3: return CXX11_clang_release_shared_capability;
    case 4: return GNU_release_generic_capability;
    case 5: return CXX11_clang_release_generic_capability;
    case 6: return GNU_unlock_function;
  }
}

static bool checkRenderScriptKernelAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkRenderScriptLangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.RenderScript)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static void matchRulesForRenderScriptKernel(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkReqdWorkGroupSizeAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkRequireConstantInitAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isGlobalVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedStaticOrTLSVar;
    return false;
  }
  return true;
}

static bool checkCPlusPlusLangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.CPlusPlus)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static void matchRulesForRequireConstantInit(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
}

static bool checkRequiresCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned RequiresCapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_requires_capability = 0,
    CXX11_clang_requires_capability = 1,
    GNU_exclusive_locks_required = 2,
    GNU_requires_shared_capability = 3,
    CXX11_clang_requires_shared_capability = 4,
    GNU_shared_locks_required = 5
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_requires_capability;
    case 1: return CXX11_clang_requires_capability;
    case 2: return GNU_exclusive_locks_required;
    case 3: return GNU_requires_shared_capability;
    case 4: return CXX11_clang_requires_shared_capability;
    case 5: return GNU_shared_locks_required;
  }
}

static bool checkRestrictAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned RestrictAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    Declspec_restrict = 0,
    GNU_malloc = 1,
    CXX11_gnu_malloc = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return Declspec_restrict;
    case 1: return GNU_malloc;
    case 2: return CXX11_gnu_malloc;
  }
}

static bool checkReturnTypestateAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrParameter;
    return false;
  }
  return true;
}

static void matchRulesForReturnTypestate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool checkReturnsNonNullAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static void matchRulesForReturnsNonNull(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkReturnsTwiceAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkScopedLockableAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool checkSectionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D) && !isGlobalVar(D) && !isa<ObjCMethodDecl>(D) && !isa<ObjCPropertyDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionGlobalVarMethodOrProperty;
    return false;
  }
  return true;
}

static unsigned SectionAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_section = 0,
    CXX11_gnu_section = 1,
    Declspec_allocate = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_section;
    case 1: return CXX11_gnu_section;
    case 2: return Declspec_allocate;
  }
}

static void matchRulesForSection(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_global, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_property, /*IsSupported=*/LangOpts.ObjC1));
}

static bool checkSetTypestateAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static void matchRulesForSetTypestate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

static bool checkSharedTrylockFunctionAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkSwiftContextAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static void matchRulesForSwiftContext(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool checkSwiftErrorResultAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static void matchRulesForSwiftErrorResult(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool checkSwiftIndirectResultAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedParameter;
    return false;
  }
  return true;
}

static void matchRulesForSwiftIndirectResult(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_parameter, /*IsSupported=*/true));
}

static bool isTLSVar(const Decl *D) {
  if (const auto *S = dyn_cast<VarDecl>(D))
    return S->getTLSKind() != 0;
  return false;
}

static bool checkTLSModelAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isTLSVar(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedTLSVar;
    return false;
  }
  return true;
}

static void matchRulesForTLSModel(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_variable_is_thread_local, /*IsSupported=*/true));
}

static bool checkTargetAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static void matchRulesForTarget(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkTestTypestateAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedMethod;
    return false;
  }
  return true;
}

static void matchRulesForTestTypestate(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
}

static bool checkThreadAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
    return false;
  }
  return true;
}

static bool checkNotCPlusPlusLangOpts(Sema &S, const AttributeList &Attr) {
  if (!S.LangOpts.CPlusPlus)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static bool checkTryAcquireCapabilityAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static unsigned TryAcquireCapabilityAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_try_acquire_capability = 0,
    CXX11_clang_try_acquire_capability = 1,
    GNU_try_acquire_shared_capability = 2,
    CXX11_clang_try_acquire_shared_capability = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_try_acquire_capability;
    case 1: return CXX11_clang_try_acquire_capability;
    case 2: return GNU_try_acquire_shared_capability;
    case 3: return CXX11_clang_try_acquire_shared_capability;
  }
}

static bool checkUnusedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<ObjCIvarDecl>(D) && !isa<TypeDecl>(D) && !isa<EnumDecl>(D) && !isa<EnumConstantDecl>(D) && !isa<LabelDecl>(D) && !isa<FieldDecl>(D) && !isa<ObjCMethodDecl>(D) && !isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedForMaybeUnused;
    return false;
  }
  return true;
}

static unsigned UnusedAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    CXX11_maybe_unused = 0,
    GNU_unused = 1,
    CXX11_gnu_unused = 2
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_maybe_unused;
    case 1: return GNU_unused;
    case 2: return CXX11_gnu_unused;
  }
}

static bool checkUuidAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D) && !isa<EnumDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedEnumOrClass;
    return false;
  }
  return true;
}

static bool checkMicrosoftExtBorlandLangOpts(Sema &S, const AttributeList &Attr) {
  if (S.LangOpts.MicrosoftExt || S.LangOpts.Borland)
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
  return false;
}

static bool checkVecReturnAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return false;
  }
  return true;
}

static bool checkVecTypeHintAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkWarnUnusedAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : ExpectedStructOrUnion);
    return false;
  }
  return true;
}

static bool checkWarnUnusedResultAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<ObjCMethodDecl>(D) && !isa<EnumDecl>(D) && !isa<CXXRecordDecl>(D) && !isFunctionLike(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodEnumOrClass;
    return false;
  }
  return true;
}

static unsigned WarnUnusedResultAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    CXX11_nodiscard = 0,
    CXX11_clang_warn_unused_result = 1,
    GNU_warn_unused_result = 2,
    CXX11_gnu_warn_unused_result = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return CXX11_nodiscard;
    case 1: return CXX11_clang_warn_unused_result;
    case 2: return GNU_warn_unused_result;
    case 3: return CXX11_gnu_warn_unused_result;
  }
}

static void matchRulesForWarnUnusedResult(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_enum, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_record, /*IsSupported=*/true));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_hasType_functionType, /*IsSupported=*/true));
}

static bool checkWeakAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D) && !isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << (S.getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass : ExpectedVariableOrFunction);
    return false;
  }
  return true;
}

static bool checkWeakRefAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return false;
  }
  return true;
}

static bool checkWorkGroupSizeHintAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return false;
  }
  return true;
}

static bool checkXRayInstrumentAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXMethodDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static unsigned XRayInstrumentAttrSpellingMap(const AttributeList &Attr) {
  enum Spelling {
    GNU_xray_always_instrument = 0,
    CXX11_clang_xray_always_instrument = 1,
    GNU_xray_never_instrument = 2,
    CXX11_clang_xray_never_instrument = 3
  };

  unsigned Idx = Attr.getAttributeSpellingListIndex();
  switch (Idx) {
    default: llvm_unreachable("Unknown spelling list index");
    case 0: return GNU_xray_always_instrument;
    case 1: return CXX11_clang_xray_always_instrument;
    case 2: return GNU_xray_never_instrument;
    case 3: return CXX11_clang_xray_never_instrument;
  }
}

static void matchRulesForXRayInstrument(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static bool checkXRayLogArgsAppertainsTo(Sema &S, const AttributeList &Attr, const Decl *D) {
  if (!isa<CXXMethodDecl>(D) && !isa<ObjCMethodDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return false;
  }
  return true;
}

static void matchRulesForXRayLogArgs(llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules, const LangOptions &LangOpts) {
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function_is_member, /*IsSupported=*/LangOpts.CPlusPlus));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_objc_method, /*IsSupported=*/LangOpts.ObjC1));
  MatchRules.push_back(std::make_pair(attr::SubjectMatchRule_function, /*IsSupported=*/true));
}

static const ParsedAttrInfo AttrInfoMap[AttributeList::UnknownAttribute + 1] = {
  { 2, 0, 0, 0, 0, 0, 0, 1, checkAMDGPUFlatWorkGroupSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAMDGPUFlatWorkGroupSize },  // AT_AMDGPUFlatWorkGroupSize
  { 1, 0, 0, 0, 0, 0, 0, 1, checkAMDGPUNumSGPRAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAMDGPUNumSGPR },  // AT_AMDGPUNumSGPR
  { 1, 0, 0, 0, 0, 0, 0, 1, checkAMDGPUNumVGPRAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAMDGPUNumVGPR },  // AT_AMDGPUNumVGPR
  { 1, 1, 0, 0, 0, 0, 0, 1, checkAMDGPUWavesPerEUAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAMDGPUWavesPerEU },  // AT_AMDGPUWavesPerEU
  { 0, 1, 1, 1, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, isTargetarmthumbarmebthumbebavrx86x86_64msp430mipsmipsel, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Interrupt
  { 0, 0, 0, 1, 0, 0, 0, 1, checkAVRSignalAppertainsTo, defaultDiagnoseLangOpts, isTargetavr, defaultSpellingIndexToSemanticSpelling, matchRulesForAVRSignal },  // AT_AVRSignal
  { 0, 15, 0, 0, 0, 0, 1, 1, checkAbiTagAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAbiTag },  // AT_AbiTag
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAcquireCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, AcquireCapabilityAttrSpellingMap, nullptr },  // AT_AcquireCapability
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAcquiredAfterAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AcquiredAfter
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAcquiredBeforeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AcquiredBefore
  { 1, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AddressSpace
  { 1, 0, 0, 0, 0, 0, 1, 0, checkAliasAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Alias
  { 1, 0, 0, 0, 0, 0, 0, 1, checkAlignValueAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAlignValue },  // AT_AlignValue
  { 0, 1, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, AlignedAttrSpellingMap, nullptr },  // AT_Aligned
  { 1, 0, 0, 0, 0, 0, 1, 0, checkAllocAlignAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AllocAlign
  { 1, 1, 0, 0, 0, 0, 1, 1, checkAllocSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAllocSize },  // AT_AllocSize
  { 0, 0, 0, 0, 0, 0, 1, 0, checkAlwaysInlineAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, AlwaysInlineAttrSpellingMap, nullptr },  // AT_AlwaysInline
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AnalyzerNoReturn
  { 1, 0, 0, 0, 0, 0, 0, 1, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAnnotate },  // AT_Annotate
  { 0, 0, 0, 1, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AnyX86NoCallerSavedRegisters
  { 0, 0, 0, 0, 0, 0, 0, 0, checkArcWeakrefUnavailableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ArcWeakrefUnavailable
  { 4, 0, 1, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, ArgumentWithTypeTagAttrSpellingMap, nullptr },  // AT_ArgumentWithTypeTag
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAssertCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, AssertCapabilityAttrSpellingMap, nullptr },  // AT_AssertCapability
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAssertExclusiveLockAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AssertExclusiveLock
  { 0, 15, 0, 0, 0, 0, 0, 0, checkAssertSharedLockAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_AssertSharedLock
  { 1, 1, 0, 0, 0, 0, 1, 1, checkAssumeAlignedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAssumeAligned },  // AT_AssumeAligned
  { 8, 0, 1, 0, 0, 0, 0, 1, checkAvailabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForAvailability },  // AT_Availability
  { 1, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Blocks
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CDecl
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCFAuditedTransferAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CFAuditedTransfer
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCFConsumedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CFConsumed
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CFReturnsNotRetained
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CFReturnsRetained
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCFUnknownTransferAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CFUnknownTransfer
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDAConstantAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDAConstant
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDADeviceAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDADevice
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDAGlobalAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDAGlobal
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDAHostAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDAHost
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDAInvalidTargetAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDAInvalidTarget
  { 1, 1, 0, 0, 0, 0, 0, 0, checkCUDALaunchBoundsAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDALaunchBounds
  { 0, 0, 0, 0, 0, 0, 0, 0, checkCUDASharedAppertainsTo, checkCUDALangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_CUDAShared
  { 0, 0, 0, 0, 0, 0, 0, 1, checkCXX11NoReturnAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForCXX11NoReturn },  // AT_CXX11NoReturn
  { 0, 15, 0, 0, 0, 0, 0, 1, checkCallableWhenAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForCallableWhen },  // AT_CallableWhen
  { 1, 0, 0, 0, 0, 0, 0, 0, checkCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, CapabilityAttrSpellingMap, nullptr },  // AT_Capability
  { 0, 0, 0, 0, 0, 0, 0, 1, checkCarriesDependencyAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForCarriesDependency },  // AT_CarriesDependency
  { 1, 0, 0, 0, 0, 0, 1, 0, checkCleanupAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Cleanup
  { 0, 0, 0, 0, 0, 0, 1, 0, checkColdAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Cold
  { 0, 0, 0, 0, 0, 0, 1, 0, checkCommonAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Common
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Const
  { 0, 1, 0, 0, 0, 0, 1, 0, checkConstructorAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Constructor
  { 1, 0, 0, 0, 0, 0, 0, 1, checkConsumableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForConsumable },  // AT_Consumable
  { 0, 0, 0, 0, 0, 0, 0, 0, checkConsumableAutoCastAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ConsumableAutoCast
  { 0, 0, 0, 0, 0, 0, 0, 0, checkConsumableSetOnReadAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ConsumableSetOnRead
  { 0, 0, 0, 0, 0, 0, 0, 1, checkConvergentAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForConvergent },  // AT_Convergent
  { 0, 0, 0, 1, 0, 0, 1, 1, checkDLLExportAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64armthumbaarch64Win32, defaultSpellingIndexToSemanticSpelling, matchRulesForDLLExport },  // AT_DLLExport
  { 0, 0, 0, 1, 0, 0, 1, 1, checkDLLImportAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64armthumbaarch64Win32, defaultSpellingIndexToSemanticSpelling, matchRulesForDLLImport },  // AT_DLLImport
  { 0, 2, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Deprecated
  { 0, 1, 0, 0, 0, 0, 1, 0, checkDestructorAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Destructor
  { 3, 0, 0, 0, 0, 0, 0, 0, checkDiagnoseIfAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_DiagnoseIf
  { 0, 0, 0, 0, 0, 0, 0, 1, checkDisableTailCallsAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForDisableTailCalls },  // AT_DisableTailCalls
  { 0, 0, 0, 1, 0, 0, 0, 0, checkEmptyBasesAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64armthumbaarch64Microsoft, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_EmptyBases
  { 2, 0, 0, 0, 0, 0, 0, 1, checkEnableIfAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForEnableIf },  // AT_EnableIf
  { 1, 0, 0, 0, 0, 0, 0, 1, checkEnumExtensibilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForEnumExtensibility },  // AT_EnumExtensibility
  { 1, 15, 0, 0, 0, 0, 0, 0, checkExclusiveTrylockFunctionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ExclusiveTrylockFunction
  { 1, 0, 0, 0, 0, 0, 0, 0, checkExtVectorTypeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ExtVectorType
  { 0, 3, 1, 0, 0, 0, 0, 1, checkExternalSourceSymbolAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForExternalSourceSymbol },  // AT_ExternalSourceSymbol
  { 0, 0, 0, 0, 0, 1, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_FallThrough
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_FastCall
  { 0, 0, 0, 0, 0, 0, 0, 1, checkFlagEnumAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForFlagEnum },  // AT_FlagEnum
  { 0, 0, 0, 0, 0, 0, 1, 1, checkFlattenAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForFlatten },  // AT_Flatten
  { 3, 0, 0, 0, 0, 0, 1, 0, checkFormatAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Format
  { 1, 0, 0, 0, 0, 0, 1, 0, checkFormatArgAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_FormatArg
  { 0, 0, 0, 0, 0, 0, 1, 0, checkGNUInlineAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_GNUInline
  { 1, 0, 0, 0, 0, 0, 0, 0, checkGuardedByAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_GuardedBy
  { 0, 0, 0, 0, 0, 0, 0, 0, checkGuardedVarAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_GuardedVar
  { 0, 0, 0, 0, 0, 0, 1, 0, checkHotAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Hot
  { 0, 0, 0, 0, 0, 0, 0, 0, checkIBActionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_IBAction
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_IBOutlet
  { 0, 1, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_IBOutletCollection
  { 1, 0, 0, 0, 0, 0, 1, 1, checkIFuncAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForIFunc },  // AT_IFunc
  { 1, 0, 0, 0, 0, 0, 0, 0, checkInitPriorityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_InitPriority
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_IntelOclBicc
  { 0, 0, 0, 0, 0, 0, 0, 1, checkInternalLinkageAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForInternalLinkage },  // AT_InternalLinkage
  { 0, 0, 0, 0, 0, 0, 0, 1, checkLTOVisibilityPublicAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForLTOVisibilityPublic },  // AT_LTOVisibilityPublic
  { 1, 0, 0, 1, 0, 0, 0, 0, checkLayoutVersionAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64armthumbaarch64Microsoft, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_LayoutVersion
  { 1, 0, 0, 0, 0, 0, 0, 0, checkLockReturnedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_LockReturned
  { 0, 0, 0, 0, 0, 0, 0, 0, checkLockableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Lockable
  { 0, 15, 0, 0, 0, 0, 0, 0, checkLocksExcludedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_LocksExcluded
  { 3, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, LoopHintAttrSpellingMap, nullptr },  // AT_LoopHint
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_MSABI
  { 0, 1, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, checkMicrosoftExtLangOpts, defaultTargetRequirements, MSInheritanceAttrSpellingMap, nullptr },  // AT_MSInheritance
  { 0, 0, 0, 1, 0, 0, 0, 0, checkMSNoVTableAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64armthumbaarch64Microsoft, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_MSNoVTable
  { 0, 0, 0, 0, 0, 0, 1, 0, checkMSStructAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_MSStruct
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_MayAlias
  { 0, 0, 0, 1, 0, 0, 1, 1, checkMicroMipsAppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipsel, defaultSpellingIndexToSemanticSpelling, matchRulesForMicroMips },  // AT_MicroMips
  { 0, 0, 0, 0, 0, 0, 0, 0, checkMinSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_MinSize
  { 0, 0, 0, 1, 0, 0, 1, 0, checkMips16AppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipsel, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Mips16
  { 0, 0, 0, 1, 0, 0, 1, 1, checkMipsLongCallAppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipselmips64mips64el, MipsLongCallAttrSpellingMap, matchRulesForMipsLongCall },  // AT_MipsLongCall
  { 0, 0, 0, 1, 0, 0, 1, 1, checkMipsShortCallAppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipselmips64mips64el, MipsShortCallAttrSpellingMap, matchRulesForMipsShortCall },  // AT_MipsShortCall
  { 1, 0, 0, 0, 0, 0, 1, 0, checkModeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Mode
  { 0, 0, 0, 0, 0, 0, 0, 0, checkNSConsumedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NSConsumed
  { 0, 0, 0, 0, 0, 0, 0, 0, checkNSConsumesSelfAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NSConsumesSelf
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NSReturnsAutoreleased
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NSReturnsNotRetained
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NSReturnsRetained
  { 0, 0, 0, 0, 0, 0, 1, 0, checkNakedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Naked
  { 1, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NeonPolyVectorType
  { 1, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NeonVectorType
  { 0, 0, 0, 0, 0, 0, 0, 0, checkNoAliasAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoAlias
  { 0, 0, 0, 0, 0, 0, 1, 0, checkNoCommonAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoCommon
  { 0, 0, 0, 0, 0, 0, 1, 1, checkNoDebugAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNoDebug },  // AT_NoDebug
  { 0, 0, 0, 0, 0, 0, 0, 1, checkNoDuplicateAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNoDuplicate },  // AT_NoDuplicate
  { 0, 0, 0, 0, 0, 0, 1, 0, checkNoInlineAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoInline
  { 0, 0, 0, 0, 0, 0, 1, 0, checkNoInstrumentFunctionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoInstrumentFunction
  { 0, 0, 0, 1, 0, 0, 1, 1, checkNoMicroMipsAppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipsel, defaultSpellingIndexToSemanticSpelling, matchRulesForNoMicroMips },  // AT_NoMicroMips
  { 0, 0, 0, 1, 0, 0, 1, 0, checkNoMips16AppertainsTo, defaultDiagnoseLangOpts, isTargetmipsmipsel, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoMips16
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoReturn
  { 0, 15, 0, 0, 0, 0, 0, 1, checkNoSanitizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNoSanitize },  // AT_NoSanitize
  { 0, 0, 0, 0, 0, 0, 1, 1, checkNoSanitizeSpecificAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNoSanitizeSpecific },  // AT_NoSanitizeSpecific
  { 0, 0, 0, 0, 0, 0, 1, 1, checkNoSplitStackAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNoSplitStack },  // AT_NoSplitStack
  { 0, 0, 0, 0, 0, 0, 0, 0, checkNoThreadSafetyAnalysisAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoThreadSafetyAnalysis
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NoThrow
  { 0, 15, 0, 0, 0, 0, 1, 0, checkNonNullAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_NonNull
  { 0, 0, 0, 0, 0, 0, 0, 1, checkNotTailCalledAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForNotTailCalled },  // AT_NotTailCalled
  { 0, 0, 0, 0, 0, 0, 0, 1, checkObjCBoxableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCBoxable },  // AT_ObjCBoxable
  { 1, 0, 0, 0, 0, 0, 0, 0, checkObjCBridgeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCBridge
  { 1, 0, 0, 0, 0, 0, 0, 0, checkObjCBridgeMutableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCBridgeMutable
  { 1, 2, 1, 0, 0, 0, 0, 0, checkObjCBridgeRelatedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCBridgeRelated
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCDesignatedInitializerAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCDesignatedInitializer
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCExceptionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCException
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCExplicitProtocolImplAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCExplicitProtocolImpl
  { 1, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCGC
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCIndependentClass
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCKindOf
  { 1, 0, 0, 0, 0, 0, 0, 1, checkObjCMethodFamilyAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCMethodFamily },  // AT_ObjCMethodFamily
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCNSObject
  { 1, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCOwnership
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCPreciseLifetimeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCPreciseLifetime
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCRequiresPropertyDefsAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCRequiresPropertyDefs
  { 0, 0, 0, 0, 0, 0, 0, 1, checkObjCRequiresSuperAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCRequiresSuper },  // AT_ObjCRequiresSuper
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCReturnsInnerPointerAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCReturnsInnerPointer
  { 0, 0, 0, 0, 0, 0, 0, 0, checkObjCRootClassAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ObjCRootClass
  { 1, 0, 0, 0, 0, 0, 0, 1, checkObjCRuntimeNameAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCRuntimeName },  // AT_ObjCRuntimeName
  { 0, 0, 0, 0, 0, 0, 0, 1, checkObjCRuntimeVisibleAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCRuntimeVisible },  // AT_ObjCRuntimeVisible
  { 0, 0, 0, 0, 0, 0, 0, 1, checkObjCSubclassingRestrictedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForObjCSubclassingRestricted },  // AT_ObjCSubclassingRestricted
  { 0, 0, 0, 0, 0, 0, 0, 0, checkOpenCLAccessAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, OpenCLAccessAttrSpellingMap, nullptr },  // AT_OpenCLAccess
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLConstantAddressSpace
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLGenericAddressSpace
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLGlobalAddressSpace
  { 1, 0, 0, 0, 0, 0, 0, 1, checkOpenCLIntelReqdSubGroupSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForOpenCLIntelReqdSubGroupSize },  // AT_OpenCLIntelReqdSubGroupSize
  { 0, 0, 0, 0, 0, 0, 0, 0, checkOpenCLKernelAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLKernel
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLLocalAddressSpace
  { 0, 0, 0, 0, 0, 0, 0, 1, checkOpenCLNoSVMAppertainsTo, checkOpenCLLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForOpenCLNoSVM },  // AT_OpenCLNoSVM
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLPrivateAddressSpace
  { 1, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_OpenCLUnrollHint
  { 0, 0, 0, 0, 0, 0, 0, 1, checkOptimizeNoneAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForOptimizeNone },  // AT_OptimizeNone
  { 0, 0, 0, 0, 0, 0, 0, 1, checkOverloadableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForOverloadable },  // AT_Overloadable
  { 1, 15, 0, 0, 0, 0, 0, 0, checkOwnershipAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, OwnershipAttrSpellingMap, nullptr },  // AT_Ownership
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Packed
  { 1, 0, 0, 0, 0, 0, 0, 1, checkParamTypestateAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForParamTypestate },  // AT_ParamTypestate
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Pascal
  { 1, 0, 0, 0, 0, 0, 0, 1, checkPassObjectSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForPassObjectSize },  // AT_PassObjectSize
  { 1, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Pcs
  { 1, 0, 0, 0, 0, 0, 0, 0, checkPragmaClangBSSSectionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PragmaClangBSSSection
  { 1, 0, 0, 0, 0, 0, 0, 0, checkPragmaClangDataSectionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PragmaClangDataSection
  { 1, 0, 0, 0, 0, 0, 0, 0, checkPragmaClangRodataSectionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PragmaClangRodataSection
  { 1, 0, 0, 0, 0, 0, 0, 0, checkPragmaClangTextSectionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PragmaClangTextSection
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PreserveAll
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PreserveMost
  { 1, 0, 0, 0, 0, 0, 0, 0, checkPtGuardedByAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PtGuardedBy
  { 0, 0, 0, 0, 0, 0, 0, 0, checkPtGuardedVarAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_PtGuardedVar
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Ptr32
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Ptr64
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Pure
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_RegCall
  { 1, 0, 0, 0, 1, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Regparm
  { 0, 15, 0, 0, 0, 0, 0, 0, checkReleaseCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, ReleaseCapabilityAttrSpellingMap, nullptr },  // AT_ReleaseCapability
  { 0, 0, 0, 0, 0, 0, 0, 1, checkRenderScriptKernelAppertainsTo, checkRenderScriptLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForRenderScriptKernel },  // AT_RenderScriptKernel
  { 3, 0, 0, 0, 0, 0, 0, 0, checkReqdWorkGroupSizeAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ReqdWorkGroupSize
  { 0, 0, 0, 0, 0, 0, 0, 1, checkRequireConstantInitAppertainsTo, checkCPlusPlusLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForRequireConstantInit },  // AT_RequireConstantInit
  { 0, 15, 0, 0, 0, 0, 0, 0, checkRequiresCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, RequiresCapabilityAttrSpellingMap, nullptr },  // AT_RequiresCapability
  { 0, 0, 0, 0, 0, 0, 1, 0, checkRestrictAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, RestrictAttrSpellingMap, nullptr },  // AT_Restrict
  { 1, 0, 0, 0, 0, 0, 0, 1, checkReturnTypestateAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForReturnTypestate },  // AT_ReturnTypestate
  { 0, 0, 0, 0, 0, 0, 1, 1, checkReturnsNonNullAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForReturnsNonNull },  // AT_ReturnsNonNull
  { 0, 0, 0, 0, 0, 0, 1, 0, checkReturnsTwiceAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ReturnsTwice
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_SPtr
  { 0, 0, 0, 0, 0, 0, 0, 0, checkScopedLockableAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ScopedLockable
  { 1, 0, 0, 0, 0, 0, 1, 1, checkSectionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, SectionAttrSpellingMap, matchRulesForSection },  // AT_Section
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_SelectAny
  { 0, 2, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Sentinel
  { 1, 0, 0, 0, 0, 0, 0, 1, checkSetTypestateAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForSetTypestate },  // AT_SetTypestate
  { 1, 15, 0, 0, 0, 0, 0, 0, checkSharedTrylockFunctionAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_SharedTrylockFunction
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_StdCall
  { 0, 15, 0, 0, 0, 1, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Suppress
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_SwiftCall
  { 0, 0, 0, 0, 0, 0, 1, 1, checkSwiftContextAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForSwiftContext },  // AT_SwiftContext
  { 0, 0, 0, 0, 0, 0, 1, 1, checkSwiftErrorResultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForSwiftErrorResult },  // AT_SwiftErrorResult
  { 0, 0, 0, 0, 0, 0, 1, 1, checkSwiftIndirectResultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForSwiftIndirectResult },  // AT_SwiftIndirectResult
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_SysVABI
  { 1, 0, 0, 0, 0, 0, 1, 1, checkTLSModelAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForTLSModel },  // AT_TLSModel
  { 1, 0, 0, 0, 0, 0, 1, 1, checkTargetAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForTarget },  // AT_Target
  { 1, 0, 0, 0, 0, 0, 0, 1, checkTestTypestateAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForTestTypestate },  // AT_TestTypestate
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_ThisCall
  { 0, 0, 0, 0, 0, 0, 0, 0, checkThreadAppertainsTo, checkMicrosoftExtLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Thread
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, checkNotCPlusPlusLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TransparentUnion
  { 1, 15, 0, 0, 0, 0, 0, 0, checkTryAcquireCapabilityAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, TryAcquireCapabilityAttrSpellingMap, nullptr },  // AT_TryAcquireCapability
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TypeNonNull
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TypeNullUnspecified
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TypeNullable
  { 4, 0, 1, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TypeTagForDatatype
  { 1, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_TypeVisibility
  { 0, 0, 0, 0, 1, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_UPtr
  { 0, 1, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Unavailable
  { 0, 0, 0, 0, 0, 0, 1, 0, checkUnusedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, UnusedAttrSpellingMap, nullptr },  // AT_Unused
  { 0, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Used
  { 1, 0, 0, 0, 0, 0, 0, 0, checkUuidAppertainsTo, checkMicrosoftExtBorlandLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Uuid
  { 0, 0, 0, 0, 0, 0, 0, 0, checkVecReturnAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_VecReturn
  { 1, 0, 0, 0, 0, 0, 0, 0, checkVecTypeHintAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_VecTypeHint
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_VectorCall
  { 1, 0, 0, 0, 1, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_VectorSize
  { 1, 0, 0, 0, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Visibility
  { 0, 0, 0, 0, 0, 0, 0, 0, checkWarnUnusedAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_WarnUnused
  { 0, 0, 0, 0, 0, 0, 1, 1, checkWarnUnusedResultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, WarnUnusedResultAttrSpellingMap, matchRulesForWarnUnusedResult },  // AT_WarnUnusedResult
  { 0, 0, 0, 0, 0, 0, 1, 0, checkWeakAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_Weak
  { 0, 0, 0, 0, 0, 0, 0, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_WeakImport
  { 0, 1, 0, 0, 0, 0, 1, 0, checkWeakRefAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_WeakRef
  { 3, 0, 0, 0, 0, 0, 0, 0, checkWorkGroupSizeHintAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_WorkGroupSizeHint
  { 0, 0, 0, 1, 0, 0, 1, 0, defaultAppertainsTo, defaultDiagnoseLangOpts, isTargetx86x86_64, defaultSpellingIndexToSemanticSpelling, nullptr },  // AT_X86ForceAlignArgPointer
  { 0, 0, 0, 0, 0, 0, 0, 1, checkXRayInstrumentAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, XRayInstrumentAttrSpellingMap, matchRulesForXRayInstrument },  // AT_XRayInstrument
  { 1, 0, 0, 0, 0, 0, 0, 1, checkXRayLogArgsAppertainsTo, defaultDiagnoseLangOpts, defaultTargetRequirements, defaultSpellingIndexToSemanticSpelling, matchRulesForXRayLogArgs }  // AT_XRayLogArgs
};

static bool checkAttributeMatchRuleAppliesTo(const Decl *D, attr::SubjectMatchRule rule) {
  switch (rule) {
  case attr::SubjectMatchRule_block:
    return isa<BlockDecl>(D);
  case attr::SubjectMatchRule_enum:
    return isa<EnumDecl>(D);
  case attr::SubjectMatchRule_enum_constant:
    return isa<EnumConstantDecl>(D);
  case attr::SubjectMatchRule_field:
    return isa<FieldDecl>(D);
  case attr::SubjectMatchRule_function:
    return isa<FunctionDecl>(D);
  case attr::SubjectMatchRule_function_is_member:
    return isa<CXXMethodDecl>(D);
  case attr::SubjectMatchRule_namespace:
    return isa<NamespaceDecl>(D);
  case attr::SubjectMatchRule_objc_category:
    return isa<ObjCCategoryDecl>(D);
  case attr::SubjectMatchRule_objc_interface:
    return isa<ObjCInterfaceDecl>(D);
  case attr::SubjectMatchRule_objc_method:
    return isa<ObjCMethodDecl>(D);
  case attr::SubjectMatchRule_objc_method_is_instance:
    return isObjCInstanceMethod(D);
  case attr::SubjectMatchRule_objc_property:
    return isa<ObjCPropertyDecl>(D);
  case attr::SubjectMatchRule_objc_protocol:
    return isa<ObjCProtocolDecl>(D);
  case attr::SubjectMatchRule_record:
    return isa<RecordDecl>(D) || isa<CXXRecordDecl>(D);
  case attr::SubjectMatchRule_record_not_is_union:
    return isStruct(D);
  case attr::SubjectMatchRule_hasType_abstract:
    assert(false && "Abstract matcher rule isn't allowed");
    return false;
  case attr::SubjectMatchRule_hasType_functionType:
    return isFunctionLike(D);
  case attr::SubjectMatchRule_type_alias:
    return isa<TypedefNameDecl>(D);
  case attr::SubjectMatchRule_variable:
    return isa<VarDecl>(D);
  case attr::SubjectMatchRule_variable_is_thread_local:
    return isTLSVar(D);
  case attr::SubjectMatchRule_variable_is_global:
    return isGlobalVar(D);
  case attr::SubjectMatchRule_variable_is_parameter:
    return isa<ParmVarDecl>(D);
  case attr::SubjectMatchRule_variable_not_is_parameter:
    return isNonParmVar(D);
  }
  llvm_unreachable("Invalid match rule");
return false;
}

