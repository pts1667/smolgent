use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{
    Attribute, Expr, ExprLit, FnArg, Ident, ItemFn, ItemStruct, Lit, Meta, Pat, PatType, Result,
    Token, Type, TypeReference, Visibility, parse_macro_input,
};

#[derive(Default)]
struct ToolArgs {
    name: Option<Expr>,
    description: Option<Expr>,
    fallible: bool,
    factory_visibility: Option<Visibility>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut args = ToolArgs::default();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "fallible" => args.fallible = true,
                "name" => {
                    input.parse::<Token![=]>()?;
                    args.name = Some(input.parse()?);
                }
                "description" => {
                    input.parse::<Token![=]>()?;
                    args.description = Some(input.parse()?);
                }
                "factory_visibility" => {
                    input.parse::<Token![=]>()?;
                    let visibility: syn::LitStr = input.parse()?;
                    args.factory_visibility = Some(visibility.parse()?);
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `name`, `description`, `fallible`, or `factory_visibility`",
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(args)
    }
}

#[proc_macro_attribute]
/// Generate a tool definition and executable factory for a Rust function.
///
/// Parameters marked `#[tool(context)]` are captured by the factory and excluded from the model
/// schema. A single `#[tool(arguments)]` parameter uses its `Deserialize + JsonSchema` type as the
/// complete model argument object. Use `fallible` when the handler returns a `Result`.
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolArgs);
    let function = parse_macro_input!(item as ItemFn);

    match expand_tool(args, function) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_tool(args: ToolArgs, mut function: ItemFn) -> Result<proc_macro2::TokenStream> {
    let function_name = function.sig.ident.clone();
    let tool_fn_name = format_ident!("{}_tool", function_name);
    let definition_fn_name = format_ident!("{}_tool_definition", function_name);
    let args_struct_name = format_ident!("__SmolgentToolArgs_{}", function_name);
    let tool_name = args
        .name
        .unwrap_or_else(|| syn::parse_quote!(stringify!(#function_name)));
    let description = args.description.unwrap_or_else(|| {
        let description = rustdoc_description(&function.attrs);
        syn::parse_quote!(#description)
    });
    let factory_visibility = args
        .factory_visibility
        .unwrap_or_else(|| function.vis.clone());
    let is_async = function.sig.asyncness.is_some();

    let mut context_parameters = Vec::new();
    let mut argument_parameter = None;
    let mut ordinary_parameters = Vec::new();

    for input in &mut function.sig.inputs {
        let FnArg::Typed(typed) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "`#[smolgent::tool]` does not support methods with `self`",
            ));
        };
        let role = take_parameter_role(&mut typed.attrs)?;
        match role {
            Some(ParameterRole::Context) => context_parameters.push(typed.clone()),
            Some(ParameterRole::Arguments) => {
                if argument_parameter.replace(typed.clone()).is_some() {
                    return Err(syn::Error::new_spanned(
                        typed,
                        "only one parameter may use `#[tool(arguments)]`",
                    ));
                }
            }
            None => ordinary_parameters.push(typed.clone()),
        }
    }

    if argument_parameter.is_some() && !ordinary_parameters.is_empty() {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "`#[tool(arguments)]` cannot be combined with ordinary model parameters",
        ));
    }

    let contexts = context_parameters
        .iter()
        .map(context_parameter)
        .collect::<Result<Vec<_>>>()?;
    let factory_parameters = contexts.iter().map(|context| {
        let name = &context.name;
        let ty = &context.owned_type;
        quote!(#name: #ty)
    });
    let clone_contexts = contexts.iter().map(|context| {
        let name = &context.name;
        quote!(let #name = #name.clone();)
    });
    let context_call_arguments = contexts.iter().map(|context| {
        let name = &context.name;
        quote!(&#name)
    });

    let (argument_type, helper_struct, model_call_arguments) =
        if let Some(parameter) = argument_parameter {
            simple_parameter_name(&parameter)?;
            let ty = parameter.ty.as_ref();
            (quote!(#ty), quote!(), vec![quote!(args)])
        } else {
            let mut fields = Vec::new();
            let mut call_arguments = Vec::new();
            for parameter in &ordinary_parameters {
                let name = simple_parameter_name(parameter)?;
                let ty = parameter.ty.as_ref();
                fields.push(quote!(pub #name: #ty));
                call_arguments.push(quote!(args.#name));
            }
            (
                quote!(#args_struct_name),
                quote! {
                    #[allow(non_camel_case_types)]
                    #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
                    struct #args_struct_name {
                        #(#fields,)*
                    }
                },
                call_arguments,
            )
        };

    let call_arguments = context_call_arguments.chain(model_call_arguments);
    let call = if is_async {
        quote!(#function_name(#(#call_arguments),*).await)
    } else {
        quote!(#function_name(#(#call_arguments),*))
    };
    let output = if args.fallible {
        quote!(let output = #call?;)
    } else {
        quote!(let output = #call;)
    };

    Ok(quote! {
        #function

        #helper_struct

        #factory_visibility fn #definition_fn_name() -> ::smolgent::ToolDefinition {
            ::smolgent::ToolDefinition::new(
                #tool_name,
                #description,
                ::schemars::schema_for!(#argument_type),
            )
        }

        #factory_visibility fn #tool_fn_name(#(#factory_parameters),*) -> ::smolgent::Tool {
            ::smolgent::Tool::new(
                #definition_fn_name(),
                move |arguments: ::serde_json::Value| {
                    #(#clone_contexts)*
                    Box::pin(async move {
                        let args: #argument_type = ::serde_json::from_value(arguments)?;
                        #output
                        Ok(::serde_json::to_value(output)?)
                    })
                },
            )
        }
    })
}

struct ContextParameter {
    name: Ident,
    owned_type: Type,
}

fn context_parameter(parameter: &PatType) -> Result<ContextParameter> {
    let name = simple_parameter_name(parameter)?;
    let Type::Reference(TypeReference {
        mutability: None,
        elem,
        ..
    }) = parameter.ty.as_ref()
    else {
        return Err(syn::Error::new_spanned(
            &parameter.ty,
            "`#[tool(context)]` parameters must be shared references such as `&AgentState`",
        ));
    };
    Ok(ContextParameter {
        name,
        owned_type: elem.as_ref().clone(),
    })
}

fn simple_parameter_name(parameter: &PatType) -> Result<Ident> {
    let Pat::Ident(pat_ident) = parameter.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &parameter.pat,
            "tool parameters must be simple identifiers",
        ));
    };
    Ok(pat_ident.ident.clone())
}

#[derive(Clone, Copy)]
enum ParameterRole {
    Context,
    Arguments,
}

fn take_parameter_role(attributes: &mut Vec<Attribute>) -> Result<Option<ParameterRole>> {
    let mut role = None;
    let mut retained = Vec::new();
    for attribute in attributes.drain(..) {
        if !attribute.path().is_ident("tool") {
            retained.push(attribute);
            continue;
        }
        let ident: Ident = attribute.parse_args()?;
        let parsed = match ident.to_string().as_str() {
            "context" => ParameterRole::Context,
            "arguments" => ParameterRole::Arguments,
            _ => {
                return Err(syn::Error::new_spanned(
                    ident,
                    "expected `context` or `arguments`",
                ));
            }
        };
        if role.replace(parsed).is_some() {
            return Err(syn::Error::new_spanned(
                attribute,
                "tool parameter role may only be specified once",
            ));
        }
    }
    *attributes = retained;
    Ok(role)
}

#[derive(Default)]
struct DefinitionArgs {
    name: Option<Expr>,
    description: Option<Expr>,
    function: Option<Ident>,
    visibility: Option<Visibility>,
}

impl Parse for DefinitionArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut args = DefinitionArgs::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "name" => args.name = Some(input.parse()?),
                "description" => args.description = Some(input.parse()?),
                "function" => {
                    let function: syn::LitStr = input.parse()?;
                    args.function = Some(function.parse()?);
                }
                "visibility" => {
                    let visibility: syn::LitStr = input.parse()?;
                    args.visibility = Some(visibility.parse()?);
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `name`, `description`, `function`, or `visibility`",
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(args)
    }
}

#[proc_macro_attribute]
/// Generate a `ToolDefinition` constructor from a `JsonSchema` argument struct.
pub fn tool_definition(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as DefinitionArgs);
    let item = parse_macro_input!(item as ItemStruct);
    match expand_tool_definition(args, item) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_tool_definition(
    args: DefinitionArgs,
    item: ItemStruct,
) -> Result<proc_macro2::TokenStream> {
    let struct_name = &item.ident;
    let name = args.name.ok_or_else(|| {
        syn::Error::new_spanned(struct_name, "`tool_definition` requires `name = ...`")
    })?;
    let description = args.description.unwrap_or_else(|| {
        let description = rustdoc_description(&item.attrs);
        syn::parse_quote!(#description)
    });
    let function = args.function.ok_or_else(|| {
        syn::Error::new_spanned(
            struct_name,
            "`tool_definition` requires `function = \"...\"`",
        )
    })?;
    let visibility = args.visibility.unwrap_or_else(|| item.vis.clone());

    Ok(quote! {
        #item

        #visibility fn #function() -> ::smolgent::ToolDefinition {
            ::smolgent::ToolDefinition::new(
                #name,
                #description,
                ::schemars::schema_for!(#struct_name),
            )
        }
    })
}

fn rustdoc_description(attrs: &[Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(name_value) if name_value.path.is_ident("doc") => {
                let Expr::Lit(ExprLit {
                    lit: Lit::Str(doc), ..
                }) = &name_value.value
                else {
                    return None;
                };
                Some(doc.value().trim().to_string())
            }
            _ => None,
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
