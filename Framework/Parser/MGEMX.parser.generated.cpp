// A Bison parser, made by GNU Bison 3.8.2.

// Skeleton implementation for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.



// First part of user prologue.
#line 2 "MGEMX.y"

#include <stdio.h>
#include <string>
#include <vector>
#include <cmath>
#include "MGEMX.scanner.generated.hpp"

#line 49 "MGEMX.parser.generated.cpp"


#include "MGEMX.parser.generated.hpp"




#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif


// Whether we are compiled with exception support.
#ifndef YY_EXCEPTIONS
# if defined __GNUC__ && !defined __EXCEPTIONS
#  define YY_EXCEPTIONS 0
# else
#  define YY_EXCEPTIONS 1
# endif
#endif



// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Symbol)         \
  do {                                          \
    if (yydebug_)                               \
    {                                           \
      *yycdebug_ << Title << ' ';               \
      yy_print_ (*yycdebug_, Symbol);           \
      *yycdebug_ << '\n';                       \
    }                                           \
  } while (false)

# define YY_REDUCE_PRINT(Rule)          \
  do {                                  \
    if (yydebug_)                       \
      yy_reduce_print_ (Rule);          \
  } while (false)

# define YY_STACK_PRINT()               \
  do {                                  \
    if (yydebug_)                       \
      yy_stack_print_ ();                \
  } while (false)

#else // !YYDEBUG

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Symbol)  YY_USE (Symbol)
# define YY_REDUCE_PRINT(Rule)           static_cast<void> (0)
# define YY_STACK_PRINT()                static_cast<void> (0)

#endif // !YYDEBUG

#define yyerrok         (yyerrstatus_ = 0)
#define yyclearin       (yyla.clear ())

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)

#line 16 "MGEMX.y"
namespace My {
#line 128 "MGEMX.parser.generated.cpp"

  /// Build a parser object.
  MGEMXParser::MGEMXParser (yyscan_t scanner_yyarg)
#if YYDEBUG
    : yydebug_ (false),
      yycdebug_ (&std::cerr),
#else
    :
#endif
      scanner (scanner_yyarg)
  {}

  MGEMXParser::~MGEMXParser ()
  {}

  MGEMXParser::syntax_error::~syntax_error () YY_NOEXCEPT YY_NOTHROW
  {}

  /*---------.
  | symbol.  |
  `---------*/

  // basic_symbol.
  template <typename Base>
  MGEMXParser::basic_symbol<Base>::basic_symbol (const basic_symbol& that)
    : Base (that)
    , value ()
  {
    switch (this->kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.copy< double > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_INT: // INT
        value.copy< long long > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.copy< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_property_list: // property_list
        value.copy< std::vector<std::string> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

  }




  template <typename Base>
  MGEMXParser::symbol_kind_type
  MGEMXParser::basic_symbol<Base>::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }


  template <typename Base>
  bool
  MGEMXParser::basic_symbol<Base>::empty () const YY_NOEXCEPT
  {
    return this->kind () == symbol_kind::S_YYEMPTY;
  }

  template <typename Base>
  void
  MGEMXParser::basic_symbol<Base>::move (basic_symbol& s)
  {
    super_type::move (s);
    switch (this->kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.move< double > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_INT: // INT
        value.move< long long > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.move< std::string > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_property_list: // property_list
        value.move< std::vector<std::string> > (YY_MOVE (s.value));
        break;

      default:
        break;
    }

  }

  // by_kind.
  MGEMXParser::by_kind::by_kind () YY_NOEXCEPT
    : kind_ (symbol_kind::S_YYEMPTY)
  {}

#if 201103L <= YY_CPLUSPLUS
  MGEMXParser::by_kind::by_kind (by_kind&& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {
    that.clear ();
  }
#endif

  MGEMXParser::by_kind::by_kind (const by_kind& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {}

  MGEMXParser::by_kind::by_kind (token_kind_type t) YY_NOEXCEPT
    : kind_ (yytranslate_ (t))
  {}



  void
  MGEMXParser::by_kind::clear () YY_NOEXCEPT
  {
    kind_ = symbol_kind::S_YYEMPTY;
  }

  void
  MGEMXParser::by_kind::move (by_kind& that)
  {
    kind_ = that.kind_;
    that.clear ();
  }

  MGEMXParser::symbol_kind_type
  MGEMXParser::by_kind::kind () const YY_NOEXCEPT
  {
    return kind_;
  }


  MGEMXParser::symbol_kind_type
  MGEMXParser::by_kind::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }



  // by_state.
  MGEMXParser::by_state::by_state () YY_NOEXCEPT
    : state (empty_state)
  {}

  MGEMXParser::by_state::by_state (const by_state& that) YY_NOEXCEPT
    : state (that.state)
  {}

  void
  MGEMXParser::by_state::clear () YY_NOEXCEPT
  {
    state = empty_state;
  }

  void
  MGEMXParser::by_state::move (by_state& that)
  {
    state = that.state;
    that.clear ();
  }

  MGEMXParser::by_state::by_state (state_type s) YY_NOEXCEPT
    : state (s)
  {}

  MGEMXParser::symbol_kind_type
  MGEMXParser::by_state::kind () const YY_NOEXCEPT
  {
    if (state == empty_state)
      return symbol_kind::S_YYEMPTY;
    else
      return YY_CAST (symbol_kind_type, yystos_[+state]);
  }

  MGEMXParser::stack_symbol_type::stack_symbol_type ()
  {}

  MGEMXParser::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
    : super_type (YY_MOVE (that.state))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.YY_MOVE_OR_COPY< double > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_INT: // INT
        value.YY_MOVE_OR_COPY< long long > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.YY_MOVE_OR_COPY< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_property_list: // property_list
        value.YY_MOVE_OR_COPY< std::vector<std::string> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

#if 201103L <= YY_CPLUSPLUS
    // that is emptied.
    that.state = empty_state;
#endif
  }

  MGEMXParser::stack_symbol_type::stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) that)
    : super_type (s)
  {
    switch (that.kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.move< double > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_INT: // INT
        value.move< long long > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.move< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_property_list: // property_list
        value.move< std::vector<std::string> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

    // that is emptied.
    that.kind_ = symbol_kind::S_YYEMPTY;
  }

#if YY_CPLUSPLUS < 201103L
  MGEMXParser::stack_symbol_type&
  MGEMXParser::stack_symbol_type::operator= (const stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.copy< double > (that.value);
        break;

      case symbol_kind::S_INT: // INT
        value.copy< long long > (that.value);
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.copy< std::string > (that.value);
        break;

      case symbol_kind::S_property_list: // property_list
        value.copy< std::vector<std::string> > (that.value);
        break;

      default:
        break;
    }

    return *this;
  }

  MGEMXParser::stack_symbol_type&
  MGEMXParser::stack_symbol_type::operator= (stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_FLT: // FLT
        value.move< double > (that.value);
        break;

      case symbol_kind::S_INT: // INT
        value.move< long long > (that.value);
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        value.move< std::string > (that.value);
        break;

      case symbol_kind::S_property_list: // property_list
        value.move< std::vector<std::string> > (that.value);
        break;

      default:
        break;
    }

    // that is emptied.
    that.state = empty_state;
    return *this;
  }
#endif

  template <typename Base>
  void
  MGEMXParser::yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const
  {
    if (yymsg)
      YY_SYMBOL_PRINT (yymsg, yysym);
  }

#if YYDEBUG
  template <typename Base>
  void
  MGEMXParser::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
  {
    std::ostream& yyoutput = yyo;
    YY_USE (yyoutput);
    if (yysym.empty ())
      yyo << "empty symbol";
    else
      {
        symbol_kind_type yykind = yysym.kind ();
        yyo << (yykind < YYNTOKENS ? "token" : "nterm")
            << ' ' << yysym.name () << " (";
        YY_USE (yykind);
        yyo << ')';
      }
  }
#endif

  void
  MGEMXParser::yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym)
  {
    if (m)
      YY_SYMBOL_PRINT (m, sym);
    yystack_.push (YY_MOVE (sym));
  }

  void
  MGEMXParser::yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym)
  {
#if 201103L <= YY_CPLUSPLUS
    yypush_ (m, stack_symbol_type (s, std::move (sym)));
#else
    stack_symbol_type ss (s, sym);
    yypush_ (m, ss);
#endif
  }

  void
  MGEMXParser::yypop_ (int n) YY_NOEXCEPT
  {
    yystack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
  MGEMXParser::debug_stream () const
  {
    return *yycdebug_;
  }

  void
  MGEMXParser::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


  MGEMXParser::debug_level_type
  MGEMXParser::debug_level () const
  {
    return yydebug_;
  }

  void
  MGEMXParser::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif // YYDEBUG

  MGEMXParser::state_type
  MGEMXParser::yy_lr_goto_state_ (state_type yystate, int yysym)
  {
    int yyr = yypgoto_[yysym - YYNTOKENS] + yystate;
    if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
      return yytable_[yyr];
    else
      return yydefgoto_[yysym - YYNTOKENS];
  }

  bool
  MGEMXParser::yy_pact_value_is_default_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yypact_ninf_;
  }

  bool
  MGEMXParser::yy_table_value_is_error_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yytable_ninf_;
  }

  int
  MGEMXParser::operator() ()
  {
    return parse ();
  }

  int
  MGEMXParser::parse ()
  {
    int yyn;
    /// Length of the RHS of the rule being reduced.
    int yylen = 0;

    // Error handling.
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// The lookahead symbol.
    symbol_type yyla;

    /// The return value of parse ().
    int yyresult;

#if YY_EXCEPTIONS
    try
#endif // YY_EXCEPTIONS
      {
    YYCDEBUG << "Starting parse\n";


    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear ();
    yypush_ (YY_NULLPTR, 0, YY_MOVE (yyla));

  /*-----------------------------------------------.
  | yynewstate -- push a new symbol on the stack.  |
  `-----------------------------------------------*/
  yynewstate:
    YYCDEBUG << "Entering state " << int (yystack_[0].state) << '\n';
    YY_STACK_PRINT ();

    // Accept?
    if (yystack_[0].state == yyfinal_)
      YYACCEPT;

    goto yybackup;


  /*-----------.
  | yybackup.  |
  `-----------*/
  yybackup:
    // Try to take a decision without lookahead.
    yyn = yypact_[+yystack_[0].state];
    if (yy_pact_value_is_default_ (yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty ())
      {
        YYCDEBUG << "Reading a token\n";
#if YY_EXCEPTIONS
        try
#endif // YY_EXCEPTIONS
          {
            yyla.kind_ = yytranslate_ (yylex (&yyla.value, scanner));
          }
#if YY_EXCEPTIONS
        catch (const syntax_error& yyexc)
          {
            YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
            error (yyexc);
            goto yyerrlab1;
          }
#endif // YY_EXCEPTIONS
      }
    YY_SYMBOL_PRINT ("Next token is", yyla);

    if (yyla.kind () == symbol_kind::S_YYerror)
    {
      // The scanner already issued an error message, process directly
      // to error recovery.  But do not keep the error token as
      // lookahead, it is too special and may lead us to an endless
      // loop in error recovery. */
      yyla.kind_ = symbol_kind::S_YYUNDEF;
      goto yyerrlab1;
    }

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.kind ();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.kind ())
      {
        goto yydefault;
      }

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
        if (yy_table_value_is_error_ (yyn))
          goto yyerrlab;
        yyn = -yyn;
        goto yyreduce;
      }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_ ("Shifting", state_type (yyn), YY_MOVE (yyla));
    goto yynewstate;


  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[+yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;


  /*-----------------------------.
  | yyreduce -- do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_ (yystack_[yylen].state, yyr1_[yyn]);
      /* Variants are always initialized to an empty instance of the
         correct type. The default '$$ = $1' action is NOT applied
         when using variants.  */
      switch (yyr1_[yyn])
    {
      case symbol_kind::S_FLT: // FLT
        yylhs.value.emplace< double > ();
        break;

      case symbol_kind::S_INT: // INT
        yylhs.value.emplace< long long > ();
        break;

      case symbol_kind::S_STR: // STR
      case symbol_kind::S_IDN: // IDN
      case symbol_kind::S_property: // property
        yylhs.value.emplace< std::string > ();
        break;

      case symbol_kind::S_property_list: // property_list
        yylhs.value.emplace< std::vector<std::string> > ();
        break;

      default:
        break;
    }



      // Perform the reduction.
      YY_REDUCE_PRINT (yyn);
#if YY_EXCEPTIONS
      try
#endif // YY_EXCEPTIONS
        {
          switch (yyn)
            {
  case 10: // namespace_declaration: NAMESPACE IDN ';'
#line 58 "MGEMX.y"
                                                    { printf("【命名空间】名称：%s\n", yystack_[1].value.as < std::string > ().c_str()); }
#line 729 "MGEMX.parser.generated.cpp"
    break;

  case 11: // enum_declaration: ENUM IDN '{' enum_value_list '}'
#line 61 "MGEMX.y"
                                                    { printf("【枚举体】名称：%s\n", yystack_[3].value.as < std::string > ().c_str()); }
#line 735 "MGEMX.parser.generated.cpp"
    break;

  case 12: // enum_declaration: ENUM IDN ':' IDN '{' enum_value_list '}'
#line 62 "MGEMX.y"
                                                    { printf("【枚举体】名称：%s ，类型：%s\n", yystack_[5].value.as < std::string > ().c_str(), yystack_[3].value.as < std::string > ().c_str()); }
#line 741 "MGEMX.parser.generated.cpp"
    break;

  case 15: // enum_value: IDN
#line 69 "MGEMX.y"
                                                    { printf("【枚举体值】%s\n", yystack_[0].value.as < std::string > ().c_str()); }
#line 747 "MGEMX.parser.generated.cpp"
    break;

  case 16: // enum_value: IDN '=' INT
#line 70 "MGEMX.y"
                                                    { printf("【枚举体值】%s = %lld\n", yystack_[2].value.as < std::string > ().c_str(), yystack_[0].value.as < long long > ()); }
#line 753 "MGEMX.parser.generated.cpp"
    break;

  case 17: // struct_declaration: STRUCT IDN '{' variable_declaration_list '}'
#line 73 "MGEMX.y"
                                                                 { 
                                                      printf("【结构体】名称：%s\n", yystack_[3].value.as < std::string > ().c_str()); }
#line 760 "MGEMX.parser.generated.cpp"
    break;

  case 20: // variable_declaration: IDN ':' IDN ';'
#line 81 "MGEMX.y"
                                                    { printf("【变量】名称：%s ，类型：%s\n", yystack_[3].value.as < std::string > ().c_str(), yystack_[1].value.as < std::string > ().c_str()); }
#line 766 "MGEMX.parser.generated.cpp"
    break;

  case 21: // variable_declaration: IDN ':' IDN property_list ';'
#line 82 "MGEMX.y"
                                                    { printf("【变量】名称：%s ，类型：%s ，%lu个属性\n", yystack_[4].value.as < std::string > ().c_str(), yystack_[2].value.as < std::string > ().c_str(), yystack_[1].value.as < std::vector<std::string> > ().size()); }
#line 772 "MGEMX.parser.generated.cpp"
    break;

  case 22: // property_list: property
#line 85 "MGEMX.y"
                                                    { yylhs.value.as < std::vector<std::string> > ().emplace_back(yystack_[0].value.as < std::string > ()); }
#line 778 "MGEMX.parser.generated.cpp"
    break;

  case 23: // property_list: property_list property
#line 86 "MGEMX.y"
                                                    { yylhs.value.as < std::vector<std::string> > ().emplace_back(yystack_[0].value.as < std::string > ()); }
#line 784 "MGEMX.parser.generated.cpp"
    break;

  case 24: // property: '(' IDN ':' STR ')'
#line 89 "MGEMX.y"
                                                    { yylhs.value.as < std::string > () = yystack_[3].value.as < std::string > () + ":" + yystack_[1].value.as < std::string > (); printf("【属性】名称：%s ，值：%s\n", yystack_[3].value.as < std::string > ().c_str(), yystack_[1].value.as < std::string > ().c_str()); }
#line 790 "MGEMX.parser.generated.cpp"
    break;

  case 25: // attribute_declaration: ATTR STR ';'
#line 92 "MGEMX.y"
                                                    { printf("【属性声明】名称：%s\n", yystack_[1].value.as < std::string > ().c_str()); }
#line 796 "MGEMX.parser.generated.cpp"
    break;

  case 26: // root_type_declaration: ROOT IDN ';'
#line 95 "MGEMX.y"
                                                    { printf("【根类型】名称：%s\n", yystack_[1].value.as < std::string > ().c_str());   }
#line 802 "MGEMX.parser.generated.cpp"
    break;

  case 27: // table_declaration: TABLE IDN '{' variable_declaration_list '}'
#line 98 "MGEMX.y"
                                                               { 
                                                      printf("【表格体】名称：%s\n", yystack_[3].value.as < std::string > ().c_str()); }
#line 809 "MGEMX.parser.generated.cpp"
    break;


#line 813 "MGEMX.parser.generated.cpp"

            default:
              break;
            }
        }
#if YY_EXCEPTIONS
      catch (const syntax_error& yyexc)
        {
          YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
          error (yyexc);
          YYERROR;
        }
#endif // YY_EXCEPTIONS
      YY_SYMBOL_PRINT ("-> $$ =", yylhs);
      yypop_ (yylen);
      yylen = 0;

      // Shift the result of the reduction.
      yypush_ (YY_NULLPTR, YY_MOVE (yylhs));
    }
    goto yynewstate;


  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_)
      {
        ++yynerrs_;
        std::string msg = YY_("syntax error");
        error (YY_MOVE (msg));
      }


    if (yyerrstatus_ == 3)
      {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        // Return failure if at end of input.
        if (yyla.kind () == symbol_kind::S_YYEOF)
          YYABORT;
        else if (!yyla.empty ())
          {
            yy_destroy_ ("Error: discarding", yyla);
            yyla.clear ();
          }
      }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and
       the label yyerrorlab therefore never appears in user code.  */
    if (false)
      YYERROR;

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();
    goto yyerrlab1;


  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;   // Each real token shifted decrements this.
    // Pop stack until we find a state that shifts the error token.
    for (;;)
      {
        yyn = yypact_[+yystack_[0].state];
        if (!yy_pact_value_is_default_ (yyn))
          {
            yyn += symbol_kind::S_YYerror;
            if (0 <= yyn && yyn <= yylast_
                && yycheck_[yyn] == symbol_kind::S_YYerror)
              {
                yyn = yytable_[yyn];
                if (0 < yyn)
                  break;
              }
          }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size () == 1)
          YYABORT;

        yy_destroy_ ("Error: popping", yystack_[0]);
        yypop_ ();
        YY_STACK_PRINT ();
      }
    {
      stack_symbol_type error_token;


      // Shift the error token.
      error_token.state = state_type (yyn);
      yypush_ ("Shifting", YY_MOVE (error_token));
    }
    goto yynewstate;


  /*-------------------------------------.
  | yyacceptlab -- YYACCEPT comes here.  |
  `-------------------------------------*/
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;


  /*-----------------------------------.
  | yyabortlab -- YYABORT comes here.  |
  `-----------------------------------*/
  yyabortlab:
    yyresult = 1;
    goto yyreturn;


  /*-----------------------------------------------------.
  | yyreturn -- parsing is finished, return the result.  |
  `-----------------------------------------------------*/
  yyreturn:
    if (!yyla.empty ())
      yy_destroy_ ("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    YY_STACK_PRINT ();
    while (1 < yystack_.size ())
      {
        yy_destroy_ ("Cleanup: popping", yystack_[0]);
        yypop_ ();
      }

    return yyresult;
  }
#if YY_EXCEPTIONS
    catch (...)
      {
        YYCDEBUG << "Exception caught: cleaning lookahead and stack\n";
        // Do not try to display the values of the reclaimed symbols,
        // as their printers might throw an exception.
        if (!yyla.empty ())
          yy_destroy_ (YY_NULLPTR, yyla);

        while (1 < yystack_.size ())
          {
            yy_destroy_ (YY_NULLPTR, yystack_[0]);
            yypop_ ();
          }
        throw;
      }
#endif // YY_EXCEPTIONS
  }

  void
  MGEMXParser::error (const syntax_error& yyexc)
  {
    error (yyexc.what ());
  }

#if YYDEBUG || 0
  const char *
  MGEMXParser::symbol_name (symbol_kind_type yysymbol)
  {
    return yytname_[yysymbol];
  }
#endif // #if YYDEBUG || 0









  const signed char MGEMXParser::yypact_ninf_ = -16;

  const signed char MGEMXParser::yytable_ninf_ = -1;

  const signed char
  MGEMXParser::yypact_[] =
  {
     -16,     0,   -16,    -9,     2,     7,    13,    10,    19,   -16,
     -16,   -16,   -16,   -16,   -16,   -16,    14,     6,    16,    17,
      20,    21,    22,    24,    27,   -16,    22,   -16,   -16,    23,
      -8,   -16,    18,     8,   -16,    26,     1,    29,   -16,   -16,
      32,   -16,    24,    24,   -16,    -5,   -16,   -16,     9,   -16,
      30,    -1,   -16,   -16,    28,   -16,   -16,    35,    25,   -16
  };

  const signed char
  MGEMXParser::yydefact_[] =
  {
       2,     0,     1,     0,     0,     0,     0,     0,     0,     3,
       4,     6,     5,     7,     8,     9,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    10,     0,    25,    26,     0,
       0,    18,    15,     0,    13,     0,     0,     0,    17,    19,
       0,    11,     0,     0,    27,     0,    16,    14,     0,    20,
       0,     0,    22,    12,     0,    21,    23,     0,     0,    24
  };

  const signed char
  MGEMXParser::yypgoto_[] =
  {
     -16,   -16,   -16,   -16,     5,     3,   -16,    31,   -15,   -16,
      -2,   -16,   -16,   -16
  };

  const signed char
  MGEMXParser::yydefgoto_[] =
  {
       0,     1,    10,    11,    33,    34,    12,    30,    31,    51,
      52,    13,    14,    15
  };

  const signed char
  MGEMXParser::yytable_[] =
  {
       2,    16,    29,     3,     4,     5,     6,     7,     8,    38,
      49,    29,    17,     9,    55,    39,    50,    18,    44,    20,
      50,    39,    23,    19,    24,    41,    53,    42,    42,    21,
      22,    25,    29,    26,    32,    27,    28,    35,    40,    45,
      54,    37,    43,    46,    58,    47,    57,    59,    48,    56,
       0,     0,     0,     0,     0,     0,     0,    36
  };

  const signed char
  MGEMXParser::yycheck_[] =
  {
       0,    10,    10,     3,     4,     5,     6,     7,     8,    17,
      15,    10,    10,    13,    15,    30,    21,    10,    17,     9,
      21,    36,    16,    10,    18,    17,    17,    19,    19,    10,
      16,    15,    10,    16,    10,    15,    15,    10,    20,    10,
      10,    18,    16,    11,     9,    42,    18,    22,    43,    51,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26
  };

  const signed char
  MGEMXParser::yystos_[] =
  {
       0,    24,     0,     3,     4,     5,     6,     7,     8,    13,
      25,    26,    29,    34,    35,    36,    10,    10,    10,    10,
       9,    10,    16,    16,    18,    15,    16,    15,    15,    10,
      30,    31,    10,    27,    28,    10,    30,    18,    17,    31,
      20,    17,    19,    16,    17,    10,    11,    28,    27,    15,
      21,    32,    33,    17,    10,    15,    33,    18,     9,    22
  };

  const signed char
  MGEMXParser::yyr1_[] =
  {
       0,    23,    24,    24,    24,    24,    24,    24,    24,    24,
      25,    26,    26,    27,    27,    28,    28,    29,    30,    30,
      31,    31,    32,    32,    33,    34,    35,    36
  };

  const signed char
  MGEMXParser::yyr2_[] =
  {
       0,     2,     0,     2,     2,     2,     2,     2,     2,     2,
       3,     5,     7,     1,     3,     1,     3,     5,     1,     2,
       4,     5,     1,     2,     5,     3,     3,     5
  };


#if YYDEBUG
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const MGEMXParser::yytname_[] =
  {
  "\"end of file\"", "error", "\"invalid token\"", "STRUCT", "ENUM",
  "NAMESPACE", "TABLE", "ATTR", "ROOT", "STR", "IDN", "INT", "FLT", "EOS",
  "EOL", "';'", "'{'", "'}'", "':'", "','", "'='", "'('", "')'", "$accept",
  "module", "namespace_declaration", "enum_declaration", "enum_value_list",
  "enum_value", "struct_declaration", "variable_declaration_list",
  "variable_declaration", "property_list", "property",
  "attribute_declaration", "root_type_declaration", "table_declaration", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const signed char
  MGEMXParser::yyrline_[] =
  {
       0,    48,    48,    49,    50,    51,    52,    53,    54,    55,
      58,    61,    62,    65,    66,    69,    70,    73,    77,    78,
      81,    82,    85,    86,    89,    92,    95,    98
  };

  void
  MGEMXParser::yy_stack_print_ () const
  {
    *yycdebug_ << "Stack now";
    for (stack_type::const_iterator
           i = yystack_.begin (),
           i_end = yystack_.end ();
         i != i_end; ++i)
      *yycdebug_ << ' ' << int (i->state);
    *yycdebug_ << '\n';
  }

  void
  MGEMXParser::yy_reduce_print_ (int yyrule) const
  {
    int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    // Print the symbols being reduced, and their result.
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
               << " (line " << yylno << "):\n";
    // The symbols being reduced.
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
                       yystack_[(yynrhs) - (yyi + 1)]);
  }
#endif // YYDEBUG

  MGEMXParser::symbol_kind_type
  MGEMXParser::yytranslate_ (int t) YY_NOEXCEPT
  {
    // YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to
    // TOKEN-NUM as returned by yylex.
    static
    const signed char
    translate_table[] =
    {
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      21,    22,     2,     2,    19,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    18,    15,
       2,    20,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    16,     2,    17,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14
    };
    // Last valid token kind.
    const int code_max = 269;

    if (t <= 0)
      return symbol_kind::S_YYEOF;
    else if (t <= code_max)
      return static_cast <symbol_kind_type> (translate_table[t]);
    else
      return symbol_kind::S_YYUNDEF;
  }

#line 16 "MGEMX.y"
} // My
#line 1195 "MGEMX.parser.generated.cpp"

#line 102 "MGEMX.y"


void My::MGEMXParser::error(const std::string& msg)
{
    std::cerr << msg << '\n';
}
