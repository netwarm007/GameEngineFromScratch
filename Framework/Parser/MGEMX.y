/* a simple material defination file parser */
%{
#include <stdio.h>
#include <string>
#include <vector>
#include <cmath>
#include "MGEMX.scanner.generated.hpp"

#include "AST.hpp"
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
%token STRUCT       /* Structure */
%token ENUM         /* Enum      */
%token NAMESPACE    /* Namespace */
%token TABLE        /* Table     */
%token ATTR         /* Attribute */
%token ROOT         /* Root Type */

%token <std::string>     STR          /* String     */
%token <std::string>     IDN          /* Identifier */
%token <long long>       INT          /* Integer    */
%token <double>          FLT          /* Float      */

%token EOS          /* End of Stream */
%token EOL          /* End of Line   */

%nterm <std::string>    property
%nterm <std::vector<std::string>>    property_list

%%
/* rules */
module: /* nothing */
    | module EOS 
    | module namespace_declaration
    | module struct_declaration 
    | module enum_declaration
    | module attribute_declaration
    | module root_type_declaration
    | module table_declaration
    ;

namespace_declaration: NAMESPACE IDN ';'            { printf("【命名空间】名称：%s\n", $2.c_str()); }
    ;

enum_declaration: ENUM IDN '{' enum_value_list '}'  { printf("【枚举体】名称：%s\n", $2.c_str()); }
    | ENUM IDN ':' IDN '{' enum_value_list '}'      { printf("【枚举体】名称：%s ，类型：%s\n", $2.c_str(), $4.c_str()); }
    ;

enum_value_list: enum_value
    | enum_value_list ',' enum_value
    ;

enum_value: IDN                                     { printf("【枚举体值】%s\n", $1.c_str()); }
    | IDN '=' INT                                   { printf("【枚举体值】%s = %lld\n", $1.c_str(), $3); }
    ;

struct_declaration: STRUCT IDN '{' variable_declaration_list '}' { 
                                                      printf("【结构体】名称：%s\n", $2.c_str()); }
    ;

variable_declaration_list: variable_declaration
    | variable_declaration_list variable_declaration
    ;

variable_declaration: IDN ':' IDN ';'               { printf("【变量】名称：%s ，类型：%s\n", $1.c_str(), $3.c_str()); }
    | IDN ':' IDN property_list ';'                 { printf("【变量】名称：%s ，类型：%s ，%lu个属性\n", $1.c_str(), $3.c_str(), $4.size()); }
    ;

property_list: property                             { $$.emplace_back($1); }
    | property_list property                        { $$.emplace_back($2); }
    ;

property: '(' IDN ':' STR ')'                       { $$ = $2 + ":" + $4; printf("【属性】名称：%s ，值：%s\n", $2.c_str(), $4.c_str()); }
    ;

attribute_declaration: ATTR STR ';'                 { printf("【属性声明】名称：%s\n", $2.c_str()); }
    ;

root_type_declaration: ROOT IDN ';'                 { printf("【根类型】名称：%s\n", $2.c_str());   }
    ;

table_declaration: TABLE IDN '{' variable_declaration_list '}' { 
                                                      printf("【表格体】名称：%s\n", $2.c_str()); }
    ;

%%

void My::MGEMXParser::error(const std::string& msg)
{
    std::cerr << msg << '\n';
}
