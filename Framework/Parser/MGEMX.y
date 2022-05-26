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

%nterm <My::ASTNodeRef> module namespace_declaration enum_declaration struct_declaration table_declaration
%nterm <std::pair<std::string, std::string>>                variable_declaration
%nterm <std::vector<std::pair<std::string, std::string>>>   variable_declaration_list
%nterm <std::string>                                        attribute
%nterm <std::vector<std::string>>                           attribute_list
%nterm <std::pair<std::string,     int32_t>>                enum_value
%nterm <std::vector<std::pair<std::string, int32_t>>>       enum_value_list

%code provides
{
    extern std::map<std::string, My::ASTNodeRef> global_symbol_table;
    extern My::ASTNodeRef ast_root;
}
%%
/* rules */
module: /* nothing */                               { 
                                                      $$ = My::make_ASTNodeRef<My::ASTNodeNone>( "MODULE" );
                                                      ast_root->SetRight($$); }
    | module EOS 
    | module namespace_declaration                  { $1->SetRight($2); $$ = $2; }
    | module enum_declaration                       { $1->SetRight($2); $$ = $2; }
    | module struct_declaration                     { $1->SetRight($2); $$ = $2; }
    | module table_declaration                      { $1->SetRight($2); $$ = $2; }
    | module attribute_declaration
    | module root_type_declaration
    ;

namespace_declaration: NAMESPACE IDN ';'            {
                                                        printf("【命名空间】名称：%s\n", $2.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeNameSpace, const char*>( 
                                                                $2.c_str(), "https://www.chenwenli.com" );
                                                    }
    ;

enum_declaration: ENUM IDN '{' enum_value_list '}'  { 
                                                        printf("【枚举体】名称：%s\n", $2.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeEnum, ASTNodeEnumValueType>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    | ENUM IDN ':' IDN '{' enum_value_list '}'      { 
                                                        printf("【枚举体】名称：%s ，类型：%s\n", $2.c_str(), $4.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeEnum, ASTNodeEnumValueType>( 
                                                                $2.c_str(), std::move($6) );
                                                    }
    ;

enum_value_list: enum_value                         {   $$  = {$1}; }
    | enum_value_list ',' enum_value                {   if(!$3.second) $3.second = $1.back().second + 1; $1.emplace_back($3); $$ = $1; }
    ;

enum_value: IDN                                     { 
                                                        printf("【枚举体值】%s\n", $1.c_str()); 
                                                        $$.first = $1;
                                                    }
    | IDN '=' INT                                   { 
                                                        printf("【枚举体值】%s = %d\n", $1.c_str(), $3); 
                                                        $$.first = $1; $$.second = $3;
                                                    }
    ;

struct_declaration: STRUCT IDN '{' variable_declaration_list '}' { 
                                                        printf("【结构体】名称：%s\n", $2.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeStruct, ASTNodeStructValueType>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    ;

variable_declaration_list: variable_declaration     {   $$ = {$1}; }
    | variable_declaration_list variable_declaration{   $1.emplace_back($2); $$ = $1; }
    ;

variable_declaration: IDN ':' IDN ';'               { 
                                                        printf("【变量】名称：%s ，类型：%s\n", $1.c_str(), $3.c_str()); 
                                                        $$.first = $1; $$.second = $3;
                                                    }
    | IDN ':' IDN attribute_list ';'                { 
                                                        printf("【变量】名称：%s ，类型：%s ，%lu个属性\n", $1.c_str(), $3.c_str(), $4.size()); 
                                                        $$.first = $1; $$.second = $3;
                                                    }
    ;

attribute_list: attribute                           { $$ = {$1}; }
    | attribute_list attribute                      { $1.emplace_back($2); $$ = $1; }
    ;

attribute: '(' IDN ':' STR ')'                      { $$ = $2 + ":" + $4; printf("【属性】名称：%s ，值：%s\n", $2.c_str(), $4.c_str()); }
    ;

attribute_declaration: ATTR STR ';'                 { printf("【属性声明】名称：%s\n", $2.c_str()); }
    ;

root_type_declaration: ROOT IDN ';'                 { printf("【根类型】名称：%s\n", $2.c_str());   }
    ;

table_declaration: TABLE IDN '{' variable_declaration_list '}' { 
                                                        printf("【表格体】名称：%s\n", $2.c_str());
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeTable, ASTNodeTableValueType>( 
                                                                $2.c_str(), std::move($4) );
                                                    }
    ;

%%

std::map<std::string, My::ASTNodeRef> global_symbol_table;

My::ASTNodeRef ast_root = My::make_ASTNodeRef<My::ASTNodeNone>( "ROOT" );

void My::MGEMXParser::error(const std::string& msg) {
    std::cerr << msg << '\n';
}

