/* a simple material defination file parser */
%{
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "AST.hpp"
#include "MGEMX.scanner.generated.hpp"

%}

%require "3.7.4"
%language "C++"
%defines "MGEMX.parser.generated.hpp"
%output  "MGEMX.parser.generated.cpp"

%define api.parser.class {MGEMXParser}
%define api.namespace {My}
%define api.value.type variant
%param {yyscan_t scanner}

%code provides
{
    #define YY_DECL \
        int yylex(My::MGEMXParser::semantic_type *yylval, yyscan_t yyscanner)
    YY_DECL;
}

/* token define */
%token                   STRUCT       /* Structure */
%token                   ENUM         /* Enum      */
%token                   NAMESPACE    /* Namespace */
%token                   TABLE        /* Table     */
%token                   ATTR         /* Attribute */
%token                   ROOT         /* Root Type */

%token <std::string>     STR          /* String     */
%token <std::string>     IDN          /* Identifier */
%token <int32_t>         INT          /* Integer    */
%token <double>          FLT          /* Float      */

%token EOS          /* End of Stream */
%token EOL          /* End of Line   */

%nterm <ASTNodeRef>   module namespace_declaration enum_declaration struct_declaration table_declaration
%nterm <ASTNodeRef>   attribute_declaration root_type_declaration
%nterm <ASTFieldDecl> field_declaration
%nterm <ASTFieldList> field_declaration_list
%nterm <std::string>  attribute
%nterm <ASTAttrList>  attribute_list
%nterm <ASTEnumItemDecl> enum_value
%nterm <ASTEnumItems> enum_value_list

%code provides
{
    namespace My {
        extern std::map<std::string, ASTNodeRef> global_symbol_table;
        extern ASTNodeRef ast_root;
    
        void register_type(ASTNodeRef);
        void register_type(ASTNode::IDN_TYPE, ASTNodeRef);
        void check_type_def(const char*);
    }
}
%%
/* rules */
module: %empty /* nothing */                        { 
                                                      $$ = make_ASTNodeRef<ASTNodeNone>( "MODULE" );
                                                      ast_root->SetRight($$); }
    | module EOS 
    | module namespace_declaration                  { register_type($2); $1->SetLeft($2); $$ = $2; }
    | module enum_declaration                       { register_type($2); $1->SetRight($2); $$ = $2; }
    | module struct_declaration                     { register_type($2); $1->SetRight($2); $$ = $2; }
    | module table_declaration                      { register_type($2); $1->SetRight($2); $$ = $2; }
    | module attribute_declaration                  { register_type($2); $1->SetRight($2); $$ = $2; }
    | module root_type_declaration                  { register_type(static_cast<ASTNode::IDN_TYPE>("[root type]"), $2); }
    ;

namespace_declaration: NAMESPACE IDN ';'            {
                                                        $$ = make_ASTNodeRef<ASTNodeNameSpace, const char*>( 
                                                                $2.c_str(), "https://www.chenwenli.com" );
                                                    }
    ;

enum_declaration: ENUM IDN '{' enum_value_list '}'  { 
                                                        $$ = make_ASTNodeRef<ASTNodeEnum, ASTEnumItems>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    | ENUM IDN ':' IDN '{' enum_value_list '}'      { 
                                                        $$ = make_ASTNodeRef<ASTNodeEnum, ASTEnumItems>( 
                                                                $2.c_str(), std::move($6) );
                                                    }
    ;

enum_value_list: enum_value                         {   $$  = {$1}; }
    | enum_value_list ',' enum_value                {   if(!$3.second) $3.second = $1.back().second + 1; $1.emplace_back($3); $$ = $1; }
    ;

enum_value: IDN                                     { 
                                                        $$.first = $1;
                                                    }
    | IDN '=' INT                                   { 
                                                        $$.first = $1; $$.second = $3;
                                                    }
    ;

struct_declaration: STRUCT IDN '{' field_declaration_list '}' { 
                                                        $$ = make_ASTNodeRef<ASTNodeStruct, ASTFieldList>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    ;

field_declaration_list: field_declaration           {   $$ = {$1}; }
    | field_declaration_list field_declaration      {   $1.emplace_back($2); $$ = $1; }
    ;

field_declaration: IDN ':' IDN ';'                  { 
                                                        auto [result, ref] = findSymbol($3.c_str());
                                                        $$.first = $1; 
                                                        if (result) $$.second = ref;
                                                    }
    | IDN ':' IDN '(' attribute_list ')' ';'        { 
                                                        auto [result, ref] = findSymbol($3.c_str());
                                                        $$.first = $1; 
                                                        if (result) $$.second = ref;
                                                    }
    ;

attribute_list: attribute                           { $$ = {$1}; }
    | attribute_list ',' attribute                  { $1.emplace_back($3); $$ = $1; }
    ;

attribute: IDN ':' STR                              { $$ = $1 + ":" + $3; }
    ;

attribute_declaration: ATTR STR ';'                 { 
                                                        $$ = make_ASTNodeRef<ASTNodeAttribute>( $2.c_str() );
                                                    }
    ;

root_type_declaration: ROOT IDN ';'                 { 
                                                        $$ = make_ASTNodeRef<ASTNodeRootType>( $2.c_str() );
                                                    }
    ;

table_declaration: TABLE IDN '{' field_declaration_list '}' { 
                                                        $$ = make_ASTNodeRef<ASTNodeTable, ASTFieldList>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    ;

%%

namespace My {

    void MGEMXParser::error(const std::string& msg) {
        std::cerr << msg << std::endl;
    }

    void register_type(ASTNode::IDN_TYPE idn, ASTNodeRef ref) {

        if(global_symbol_table.end() != global_symbol_table.find(idn)) {
            // symbol already defined
            std::cerr << "\x1b[41m\x1b[33m";
            std::cerr << "【错误】符号重定义 {" << idn << "} ！" << std::endl;
            std::cerr << "\x1b[49m\x1b[37m";
        }
        else {
            global_symbol_table.emplace(idn, ref);
            std::cerr << "\x1b[46m\x1b[30m";
            std::cerr << "【信息】符号{" << idn << "}已经注册到全局符号表！" << std::endl;
            std::cerr << "\x1b[49m\x1b[37m";
        }
    }

    void register_type(ASTNodeRef ref) {
        assert(ref);
        register_type(ref->GetIDN(), ref);
    }

    void check_type_def(const char* idn) {
        if(global_symbol_table.end() == global_symbol_table.find(idn)) {
            // symbol type is not defined yet
            std::cerr << "\x1b[41m\x1b[33m";
            std::cerr << "【错误】类型{" << idn << "}尚未定义！" << std::endl;
            std::cerr << "\x1b[49m\x1b[37m";
        }
    }

} // namespace My
