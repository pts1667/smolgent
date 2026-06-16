use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{
    Attribute, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Meta, Pat, Result, Token, Type,
    Visibility, parse_macro_input,
};

#[derive(Default)]
struct ToolArgs {
    name: Option<String>,
    description: Option<String>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut args = ToolArgs::default();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: syn::LitStr = input.parse()?;

            match key.to_string().as_str() {
                "name" => args.name = Some(value.value()),
                "description" => args.description = Some(value.value()),
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `name` or `description`",
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
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolArgs);
    let function = parse_macro_input!(item as ItemFn);

    match expand_tool(args, function) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_tool(args: ToolArgs, function: ItemFn) -> Result<proc_macro2::TokenStream> {
    let function_name = function.sig.ident.clone();
    let tool_fn_name = format_ident!("{}_tool", function_name);
    let args_struct_name = format_ident!("__SmolgentToolArgs_{}", function_name);
    let tool_name = args.name.unwrap_or_else(|| function_name.to_string());
    let description = args
        .description
        .unwrap_or_else(|| rustdoc_description(&function.attrs));
    let visibility = function.vis.clone();
    let is_async = function.sig.asyncness.is_some();

    let mut arg_fields = Vec::new();
    let mut call_args = Vec::new();
    for input in &function.sig.inputs {
        let FnArg::Typed(typed) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "`#[smolgent::tool]` does not support methods with `self`",
            ));
        };

        let Pat::Ident(pat_ident) = typed.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "tool arguments must be simple identifiers",
            ));
        };

        let arg_name = pat_ident.ident.clone();
        let ty: Type = typed.ty.as_ref().clone();
        arg_fields.push(quote! {
            pub #arg_name: #ty
        });
        call_args.push(quote! {
            args.#arg_name
        });
    }

    let call = if is_async {
        quote! { #function_name(#(#call_args),*).await }
    } else {
        quote! { #function_name(#(#call_args),*) }
    };

    let helper_visibility = helper_visibility(&visibility);

    Ok(quote! {
        #function

        #[allow(non_camel_case_types)]
        #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
        #helper_visibility struct #args_struct_name {
            #(#arg_fields,)*
        }

        #visibility fn #tool_fn_name() -> ::smolgent::Tool {
            ::smolgent::Tool::new(
                ::smolgent::ToolDefinition::new(
                    #tool_name,
                    #description,
                    ::schemars::schema_for!(#args_struct_name),
                ),
                |arguments: ::serde_json::Value| {
                    Box::pin(async move {
                        let args: #args_struct_name = ::serde_json::from_value(arguments)?;
                        let output = #call;
                        Ok(::serde_json::to_value(output)?)
                    })
                },
            )
        }
    })
}

fn helper_visibility(visibility: &Visibility) -> proc_macro2::TokenStream {
    match visibility {
        Visibility::Public(_) => quote! { pub },
        _ => quote! {},
    }
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
