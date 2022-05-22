/* a simple material defination file parser */
%{
#include <stdio.h>
#include <string>
#include <cmath>
#include "MGEMX.scanner.hpp"
%}

%require "3.7.4"
%language "C++"
%defines "MGEMX.parser.hpp"
%output  "MGEMX.parser.cpp"

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
%token OP           /* Open Parenthesis  '(' */
%token CP           /* Close Parenthesis ')' */
%token COLON        /* ':' */
%token SEMIC        /* ';' */
%token ASTER        /* '*' */
%token COMMA        /* ',' */
%token OBK          /* Open Bracket  '[' */
%token CBK          /* Close Bracket ']' */
%token OBR          /* Open Brace    '{' */
%token CBR          /* Close Brace   '}' */
%token OAB          /* Open Angle Bracket  '<' */
%token CAB          /* Close Angle Bracket '>' */

%token <std::string>     STR          /* String     */
%token <std::string>     IDN          /* Identifier */
%token <long long>       INT          /* Integer    */
%token <double>          FLT          /* Float      */

%token EOS          /* End of Stream */
%token EOL          /* End of Line   */

%%
/* rules */
module: /* nothing */
    | module EOS 
    | module struct_declaration 
    | module enum_declaration
    ;

variable_declaration: IDN IDN SEMIC { printf("【定义变量】名称：%s ，类型：%s\n", $2.c_str(), $1.c_str()); }
    ;

variable_declaration_list: variable_declaration
    | variable_declaration_list variable_declaration
    ;

struct_declaration: IDN IDN OBR variable_declaration_list CBR SEMIC { printf("【定义结构体】名称：%s\n", $2.c_str()); }
    ;

enum_declaration: IDN IDN OBR enum_value_list CBR { printf("【定义枚举体】名称：%s\n", $2.c_str()); }
    | IDN IDN COLON IDN OBR enum_value_list CBR   { printf("【定义显式指定类型的枚举体】名称：%s ，类型：%s\n", $2.c_str(), $4.c_str()); }
    ;

enum_value_list: enum_value
    | enum_value_list COMMA enum_value
    ;

enum_value: IDN         { printf("【枚举体值】%s\n", $1.c_str()); }
    ;

%%

void My::MGEMXParser::error(const std::string& msg)
{
    std::cerr << msg << '\n';
}