from __future__ import annotations
from typing import Annotated, Any
import jax
import jax.numpy as jnp
import jax.random as jr


def sample_task(
        examples: Annotated[jax.Array, "(num_examples, 2, nx)"],
        key: Annotated[jax.Array, "() | (2,)"],
        *,
        k: int,
        ) -> tuple[
            Annotated[jax.Array, "(k, 2, nx)"],
            Annotated[jax.Array, "(2, nx)"]
            ]:
    """
    sample task from examples \\
    task{ \\
    context = examples[0], examples[10], examples[20], examples[30] \\
    query = examples[40] \\
    }

    parameters
    ----------
    examples
        input-target examples
    key
        random number generator
    k
        number of context examples

    returns
    -------
    context
        input-target examples
    query
        held-out input-target example
    """
    # TODO: parameterize the number of query examples
    indices = jr.choice(key, len(examples), shape=(k + 1,), replace=False)
    context = jnp.array([examples[i] for i in indices[:-1]])
    query = jnp.array(examples[indices[-1]])
    return context, query


def tokenize(
        x: Annotated[jax.Array, "(nx,)"],
        context: Annotated[jax.Array, "(k, 2, nx)"],
        query: Annotated[jax.Array, "(2, nx)"],
        ) -> tuple[
            Annotated[jax.Array, "(num_input_tokens, token_dim)"],
            Annotated[jax.Array, "(num_target_tokens, token_dim)"]
            ]:
    """
    convert function values into sequence values

    parameters
    ----------
    x
        spatial coordinates
    context
        input-target examples
    query
        held-out input-target example

    returns
    -------
    input_tokens
        sequence of tokenized input function values
    target_tokens
        sequence of tokenized target function values
    """
    input_tokens = list()
    for i in range(context.shape[0]):  # example index
        for j in range(context.shape[1]):  # input-target index
            for k in range(context.shape[2]):  # spatial index
                role = None
                if j == 0:
                    role = 0  # context input
                else:
                    role = 1  # context target
                x_k = x[k]
                tok = (x_k, context[i][j][k], role)
                input_tokens.append(tok)
    target_tokens = list()
    for i in range(len(query)):  # input-target index
        for j in range(len(query[0])):  # spatial index
            x_j = x[j]
            if i == 0:
                role = 2  # query input
                tok = (x_j, query[i][j], role)
                input_tokens.append(tok)
            else:
                role = 3  # query target
                tok = (x_j, query[i][j], role)
                target_tokens.append(tok)
    input_tokens = jnp.array(input_tokens)
    target_tokens = jnp.array(target_tokens)
    return input_tokens, target_tokens


def batch_task(
        x: Annotated[jax.Array, "(nx,)"],
        examples: Annotated[jax.Array, "(num_examples, 2, nx)"],
        key: Annotated[jax.Array, "() | (2,)"],
        *,
        n: int,
        k: int,
        ) -> tuple[
            Annotated[jax.Array, "(n, num_input_tokens, token_dim)"],
            Annotated[jax.Array, "(n, num_target_tokens, token_dim)"]
            ]:
    """
    create batch of tokenized tasks

    parameters
    ----------
    x
        spatial coordinates
    examples
        input-target examples
    key
        random number generator
    n
        number of sampled tasks
    k
        number of context examples

    returns
    -------
    input_batch
        sampled input tasks
    target_batch
        sampled target tasks
    """
    input_batch = list()
    target_batch = list()
    for i in range(n):
        key, subkey = jr.split(key)
        context, query = sample_task(examples, subkey, k=k)
        input_tokens, target_tokens = tokenize(x, context, query)
        input_batch.append(input_tokens)
        target_batch.append(target_tokens)
    input_batch = jnp.array(input_batch)
    target_batch = jnp.array(target_batch)
    return input_batch, target_batch


def valid_task(
        input_batch: Annotated[jax.Array, "(n, num_input_tokens, token_dim)"],
        target_batch: Annotated[jax.Array, "(n, num_target_tokens, token_dim)"],
        *,
        k: int,
        nx: int
        ) -> tuple[
            bool,
            dict[str, Any],
            ]:
    """
    validate batch of tokenized tasks

    parameters
    ----------
    input_batch:
        sampled input tasks
    target_batch:
        sampled target tasks
    k:
        number of context examples
    nx:
        number of spatial coordinates

    returns
    -------
    is_valid:
        boolean indicating whether the batch is valid or not
    report:
        validation summary
    """
    return None


def main() -> None:
    pass

if __name__ == "__main__":
    main()
