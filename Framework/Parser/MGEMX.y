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
    namespace My {
        extern std::map<std::string, ASTNodeRef> global_symbol_table;
        extern ASTNodeRef ast_root;
    
        void register_type(const char*, ASTNodeRef);
        void check_type_def(const char*);
    }
}
%%
/* rules */
module: %empty /* nothing */                        { 
                                                      $$ = My::make_ASTNodeRef<My::ASTNodeNone>( "MODULE" );
                                                      ast_root->SetRight($$); }
    | module EOS 
    | module namespace_declaration                  { $1->SetLeft($2); $$ = $2; }
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
                                                        My::register_type($2.c_str(), $$);
                                                    }
    ;

enum_declaration: ENUM IDN '{' enum_value_list '}'  { 
                                                        printf("【枚举体】名称：%s\n", $2.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeEnum, ASTNodeEnumValueType>( 
                                                                $2.c_str(), std::move($4) );
                                                        My::register_type($2.c_str(), $$);
                                                    }
    | ENUM IDN ':' IDN '{' enum_value_list '}'      { 
                                                        printf("【枚举体】名称：%s ，类型：%s\n", $2.c_str(), $4.c_str()); 
                                                        $$ = My::make_ASTNodeRef<My::ASTNodeEnum, ASTNodeEnumValueType>( 
                                                                $2.c_str(), std::move($6) );
                                                        My::register_type($2.c_str(), $$);
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
                                                        My::register_type($2.c_str(), $$);
                                                    }
    ;

variable_declaration_list: variable_declaration     {   $$ = {$1}; }
    | variable_declaration_list variable_declaration{   $1.emplace_back($2); $$ = $1; }
    ;

variable_declaration: IDN ':' IDN ';'               { 
                                                        printf("【变量】名称：%s ，类型：%s\n", $1.c_str(), $3.c_str()); 
                                                        My::check_type_def($3.c_str());
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
                                                        My::register_type($2.c_str(), $$);
                                                    }
    ;

%%

namespace My {

void MGEMXParser::error(const std::string& msg) {
    std::cerr << msg << std::endl;
}

void register_type(const char* idn, ASTNodeRef ref) {

    if(global_symbol_table.end() != global_symbol_table.find(idn)) {
        // symbol already defined
        std::cerr << "\x1b[41m\x1b[33m";
        std::cerr << "【错误】符号重定义 {" << idn << "} ！" << std::endl;
        std::cerr << "\x1b[49m\x1b[37m";
    }
    else {
        global_symbol_table.emplace(idn, ref);
        std::cout << "\x1b[46m\x1b[30m";
        std::cout << "【信息】符号{" << idn << "}已经注册到全局符号表！" << std::endl;
        std::cout << "\x1b[49m\x1b[37m";
    }
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
