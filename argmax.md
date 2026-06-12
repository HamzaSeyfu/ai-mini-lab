# Contents

* **ArgMax** operator for type [real](#real)
* **ArgMax** operator for types [float16, float, double](#float)
* **ArgMax** operator for integer types [int8, int16, int32, int64, uint8, uint16, uint32, uint64](#integer)

Based on ONNX documentation [ArgMax version 13](https://onnx.ai/onnx/operators/onnx__ArgMax.html).

<a id="real"></a>

# **ArgMax** (real)

## Signature

$Y = \textbf{ArgMax}(X)$

where:

* $X$: input tensor
* $Y$: output tensor containing the indices of the maximum values of $X$ along a given axis

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

| Restriction            | Statement                                                   | Origin                                                                   |
| ---------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| `[R1]` <a id="R1"></a> | Attribute `axis` must be set                                | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R2]` <a id="R2"></a> | Attribute `keepdims` must be set                            | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R3]` <a id="R3"></a> | Attribute `select_last_index` must be set                   | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R4]` <a id="R4"></a> | `axis` $\ge 0$                                              | Transient                                                                |
| `[R5]` <a id="R5"></a> | The dimension of $X$ along `axis` must be strictly positive | Mathematical definition                                                  |

---

## Informal specification

The **ArgMax** operator computes the index of the maximum value of the input tensor $X$ along a specified axis.

Let:

* $r$ be the rank of $X$,
* $\textit{axis}$ be the attribute specifying the dimension along which the maximum is searched,
* $dX_k$ be the size of $X$ along dimension $k$,
* $dX_{axis}$ be the size of $X$ along the reduced dimension,
* $\textit{keepdims}$ specify whether the reduced dimension is kept in the output shape,
* $\textit{select_last_index}$ specify whether the first or last index is selected when the maximum value appears more than once.

For any fixed indices outside the `axis` dimension, the operator searches the maximum value among all elements:

$$
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
$$

where:

$$
0 \le j < dX_{axis}
$$

Let:

$$
M(i_0,\dots,i_{axis-1},i_{axis+1},\dots,i_{r-1})
================================================

\max_{0 \le j < dX_{axis}}
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
$$

The output value is the index $j$ where this maximum is reached.

If `select_last_index = 0`, the first occurrence of the maximum value is selected:

$$
Y[k] =
\min
\left{
j \mid
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
==================================================

M(i_0,\dots,i_{axis-1},i_{axis+1},\dots,i_{r-1})
\right}
$$

If `select_last_index = 1`, the last occurrence of the maximum value is selected:

$$
Y[k] =
\max
\left{
j \mid
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
==================================================

M(i_0,\dots,i_{axis-1},i_{axis+1},\dots,i_{r-1})
\right}
$$

where $k$ is the output tensor index corresponding to the fixed indices outside the reduced axis.

The shape of the output tensor depends on the value of `keepdims`.

If `keepdims = 1`, the output tensor keeps the same rank as $X$, and the reduced dimension is replaced by $1$:

$$
shape(Y) =
(dX_0,\dots,dX_{axis-1},1,dX_{axis+1},\dots,dX_{r-1})
$$

If `keepdims = 0`, the reduced dimension is removed:

$$
shape(Y) =
(dX_0,\dots,dX_{axis-1},dX_{axis+1},\dots,dX_{r-1})
$$

The output tensor contains integer indices in the range:

$$
0 \le Y[k] < dX_{axis}
$$

---

### Example 1: 1D tensor

```math
X = \begin{bmatrix} 1 & 5 & 3 & 2 \end{bmatrix}
```

with:

```math
axis = 0,\quad keepdims = 1,\quad select\_last\_index = 0
```

The maximum value is $5$, located at index $1$.

```math
Y = \begin{bmatrix} 1 \end{bmatrix}
```

---

### Example 2: 2D tensor with `axis = 0`

```math
X =
\begin{bmatrix}
1 & 9 & 3 \\
4 & 2 & 6
\end{bmatrix}
```

with:

```math
axis = 0,\quad keepdims = 1,\quad select\_last\_index = 0
```

The maximum is computed column by column.

```math
Y =
\begin{bmatrix}
1 & 0 & 1
\end{bmatrix}
```

Explanation:

* column 0: $\max(1,4)=4$, index $1$
* column 1: $\max(9,2)=9$, index $0$
* column 2: $\max(3,6)=6$, index $1$

---

### Example 3: 2D tensor with `axis = 1`

```math
X =
\begin{bmatrix}
1 & 9 & 3 \\
4 & 2 & 6
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

The maximum is computed row by row.

```math
Y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

Explanation:

* row 0: $\max(1,9,3)=9$, index $1$
* row 1: $\max(4,2,6)=6$, index $2$

---

### Example 4: `keepdims = 0`

```math
X =
\begin{bmatrix}
1 & 9 & 3 \\
4 & 2 & 6
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 0,\quad select\_last\_index = 0
```

The maximum is computed row by row, and the reduced dimension is removed.

```math
Y =
\begin{bmatrix}
1 & 2
\end{bmatrix}
```

The shape of $X$ is $(2,3)$ and the shape of $Y$ is $(2)$.

---

### Example 5: repeated maximum with first index selected

```math
X =
\begin{bmatrix}
1 & 5 & 5 & 2
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

The maximum value is $5$, appearing at indices $1$ and $2$ along `axis`.

Since `select_last_index = 0`, the first occurrence is selected.

```math
Y =
\begin{bmatrix}
1
\end{bmatrix}
```

---

### Example 6: repeated maximum with last index selected

```math
X =
\begin{bmatrix}
1 & 5 & 5 & 2
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 1
```

The maximum value is $5$, appearing at indices $1$ and $2$ along `axis`.

Since `select_last_index = 1`, the last occurrence is selected.

```math
Y =
\begin{bmatrix}
2
\end{bmatrix}
```

---

### Example 7: 3D tensor

```math
X =
\begin{bmatrix}
\begin{bmatrix}
1 & 3 & 2 \\
4 & 0 & 6
\end{bmatrix}
\\
\begin{bmatrix}
7 & 5 & 9 \\
2 & 8 & 1
\end{bmatrix}
\end{bmatrix}
```

Shape:

```math
shape(X) = (2,2,3)
```

with:

```math
axis = 2,\quad keepdims = 1,\quad select\_last\_index = 0
```

The maximum is computed along the last dimension.

```math
Y =
\begin{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix}
\\
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\end{bmatrix}
```

Shape:

```math
shape(Y) = (2,2,1)
```

---

## Error conditions

The operator is undefined if one of the following conditions holds:

* `axis < 0`
* `axis >= r`, where $r$ is the rank of $X$
* $dX_{axis} = 0$
* `keepdims` is not equal to `0` or `1`
* `select_last_index` is not equal to `0` or `1`

## Attributes

### `axis`: int

Specifies the dimension along which the maximum index is computed.

#### Constraints

* `[C1]` Value domain

  * Statement: `axis >= 0`. `[R4]`

* `[C2]` Consistency with tensor rank

  * Statement: `axis < r`, where $r$ is the rank of $X$.

### `keepdims`: int

Specifies whether the reduced dimension is kept in the output tensor.

If `keepdims = 1`, the reduced dimension is kept with size $1$.

If `keepdims = 0`, the reduced dimension is removed.

#### Constraints

* `[C1]` Value domain

  * Statement: `keepdims` must be equal to `0` or `1`.

### `select_last_index`: int

Specifies how ties are resolved when the maximum value appears more than once along the reduced axis.

If `select_last_index = 0`, the first index of the maximum value is selected.

If `select_last_index = 1`, the last index of the maximum value is selected.

#### Constraints

* `[C1]` Value domain

  * Statement: `select_last_index` must be equal to `0` or `1`.

## Inputs

### $\text{X}$: real tensor

Input tensor.

#### Constraints

* `[C1]` <a id="C1rx"></a> Axis consistency

  * Statement: The rank $r$ of $X$ shall satisfy `axis < r`.

* `[C2]` <a id="C2rx"></a> Non-empty reduced dimension

  * Statement: The dimension of $X$ along `axis` shall be strictly positive.

## Outputs

### $\text{Y}$: int64 tensor

Output tensor containing the indices of the maximum values of $X$ along `axis`.

#### Constraints

* `[C1]` <a id="C1ry"></a> Output type

  * Statement: $Y$ shall have type `int64`.

* `[C2]` <a id="C2ry"></a> Output shape consistency

  * Statement: If `keepdims = 1`, then:

```math
shape(Y) =
(dX_0,\dots,dX_{axis-1},1,dX_{axis+1},\dots,dX_{r-1})
```

* Statement: If `keepdims = 0`, then:

```math
shape(Y) =
(dX_0,\dots,dX_{axis-1},dX_{axis+1},\dots,dX_{r-1})
```

* `[C3]` <a id="C3ry"></a> Output value range

  * Statement: For every output index $k$:

```math
0 \le Y[k] < dX_{axis}
```

<a id="float"></a>

# **ArgMax** (float)

where float is in {float16, float, double}.

## Signature

$Y = \textbf{ArgMax}(X)$

where:

* $X$: floating-point input tensor
* $Y$: int64 output tensor containing the indices of the maximum values of $X$ along a given axis

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

See [Restrictions](#real).

---

## Informal specification

The **ArgMax** operator computes the index of the maximum floating-point value of the input tensor $X$ along a specified axis according to IEEE 754 floating-point semantics.

For finite values, the behavior is the same as for real numbers.

For any fixed indices outside the `axis` dimension, the operator searches the index of the maximum value among:

```math
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
```

where:

```math
0 \le j < dX_{axis}
```

If the reduced slice contains no `NaN`, then:

* `+inf` is considered greater than all finite values,
* finite values are ordered according to the usual floating-point order,
* `-inf` is considered smaller than all finite values,
* if the maximum appears several times, the selected index depends on `select_last_index`.

If the reduced slice contains one or more `NaN` values, the result is defined as follows in order to keep the operator deterministic:

* if `select_last_index = 0`, the index of the first `NaN` along the reduced axis is returned,
* if `select_last_index = 1`, the index of the last `NaN` along the reduced axis is returned.

This convention gives priority to `NaN` values over finite and infinite values in the reduced slice.

More formally, for any reduced slice:

```math
S =
\left\{
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
\mid
0 \le j < dX_{axis}
\right\}
```

If there exists at least one index $j$ such that:

```math
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}] = \text{NaN}
```

then:

```math
Y[k] =
\begin{cases}
\min \left\{ j \mid X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}] = \text{NaN} \right\}
& \text{if } select\_last\_index = 0 \\

\max \left\{ j \mid X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}] = \text{NaN} \right\}
& \text{if } select\_last\_index = 1
\end{cases}
```

Otherwise, let:

```math
M =
\max_{0 \le j < dX_{axis}}
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
```

Then:

```math
Y[k] =
\begin{cases}
\min
\left\{
j \mid
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
=
M
\right\}
& \text{if } select\_last\_index = 0 \\

\max
\left\{
j \mid
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
=
M
\right\}
& \text{if } select\_last\_index = 1
\end{cases}
```

---

### Example 1: finite floating-point values

```math
X =
\begin{bmatrix}
1.0 & 9.5 & 3.0 \\
4.0 & 2.0 & 6.5
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

---

### Example 2: `+inf`

```math
X =
\begin{bmatrix}
1.0 & +\inf & 3.0 \\
4.0 & 2.0 & 6.5
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

Explanation:

* row 0: `+inf` is the maximum, index $1$
* row 1: $6.5$ is the maximum, index $2$

---

### Example 3: `-inf`

```math
X =
\begin{bmatrix}
-\inf & -4.0 & -2.0 \\
-\inf & -\inf & -\inf
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

Explanation:

* row 0: $-2.0$ is the maximum, index $2$
* row 1: all values are equal to $-\inf$, so the first index is selected

---

### Example 4: `NaN` with first index selected

```math
X =
\begin{bmatrix}
1.0 & \text{NaN} & 3.0 \\
4.0 & 2.0 & 6.5
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

Explanation:

* row 0 contains `NaN`, so the first `NaN` index is selected
* row 1 contains no `NaN`, so the maximum finite value is selected

---

### Example 5: `NaN` with last index selected

```math
X =
\begin{bmatrix}
\text{NaN} & 1.0 & \text{NaN} & 3.0
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 1
```

```math
Y =
\begin{bmatrix}
2
\end{bmatrix}
```

Explanation:

The row contains two `NaN` values at indices $0$ and $2$. Since `select_last_index = 1`, the last `NaN` index is selected.

---

### Example 6: repeated maximum with floating-point values

```math
X =
\begin{bmatrix}
1.0 & 7.0 & 7.0 & 2.0
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 1
```

```math
Y =
\begin{bmatrix}
2
\end{bmatrix}
```

---

## Error conditions

The operator is undefined if one of the following conditions holds:

* `axis < 0`
* `axis >= r`, where $r$ is the rank of $X$
* $dX_{axis} = 0$
* `keepdims` is not equal to `0` or `1`
* `select_last_index` is not equal to `0` or `1`

The operator does not return `NaN`, because the output tensor contains integer indices.

## Attributes

### `axis`: int

See [ArgMax (real)](#real).

### `keepdims`: int

See [ArgMax (real)](#real).

### `select_last_index`: int

See [ArgMax (real)](#real).

## Inputs

### $\text{X}$: floating-point tensor

Input tensor.

#### Constraints

* `[C1]` <a id="C1fx"></a> Axis consistency

  * Statement: The rank $r$ of $X$ shall satisfy `axis < r`.

* `[C2]` <a id="C2fx"></a> Non-empty reduced dimension

  * Statement: The dimension of $X$ along `axis` shall be strictly positive.

## Outputs

### $\text{Y}$: int64 tensor

Output tensor containing the indices of the maximum values of $X$ along `axis`.

#### Constraints

* `[C1]` <a id="C1fy"></a> Output type

  * Statement: $Y$ shall have type `int64`.

* `[C2]` <a id="C2fy"></a> Output shape consistency

  * Statement: See constraint [C2](#C2ry) on tensor $Y$ in [ArgMax (real)](#real).

* `[C3]` <a id="C3fy"></a> Output value range

  * Statement: See constraint [C3](#C3ry) on tensor $Y$ in [ArgMax (real)](#real).

## Numeric accuracy

No numeric accuracy note is required for the output, since **ArgMax** returns integer indices.

The only floating-point aspects concern value comparison along the reduced axis, including the handling of `NaN`, `+inf`, and `-inf`.

<a id="integer"></a>

# **ArgMax** (integer)

where integer is in {int8, int16, int32, int64, uint8, uint16, uint32, uint64}.

## Signature

$Y = \textbf{ArgMax}(X)$

where:

* $X$: integer input tensor
* $Y$: int64 output tensor containing the indices of the maximum values of $X$ along a given axis

## Restrictions

[General restrictions](./../common/general_restrictions.md) are applicable.

See [Restrictions](#real).

---

## Informal specification

The **ArgMax** operator computes the index of the maximum integer value of the input tensor $X$ along a specified axis.

For any fixed indices outside the `axis` dimension, the operator searches the maximum value among:

```math
X[i_0,\dots,i_{axis-1},j,i_{axis+1},\dots,i_{r-1}]
```

where:

```math
0 \le j < dX_{axis}
```

If the maximum value appears several times, the selected index depends on `select_last_index`.

If `select_last_index = 0`, the first occurrence of the maximum value is selected.

If `select_last_index = 1`, the last occurrence of the maximum value is selected.

The output tensor contains indices of type `int64`.

---

### Example 1: integer tensor

```math
X =
\begin{bmatrix}
1 & 9 & 3 \\
4 & 2 & 6
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

---

### Example 2: repeated maximum

```math
X =
\begin{bmatrix}
2 & 8 & 8 & 1
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 0
```

```math
Y =
\begin{bmatrix}
1
\end{bmatrix}
```

with:

```math
axis = 1,\quad keepdims = 1,\quad select\_last\_index = 1
```

```math
Y =
\begin{bmatrix}
2
\end{bmatrix}
```

---

## Error conditions

The operator is undefined if one of the following conditions holds:

* `axis < 0`
* `axis >= r`, where $r$ is the rank of $X$
* $dX_{axis} = 0$
* `keepdims` is not equal to `0` or `1`
* `select_last_index` is not equal to `0` or `1`

## Attributes

### `axis`: int

See [ArgMax (real)](#real).

### `keepdims`: int

See [ArgMax (real)](#real).

### `select_last_index`: int

See [ArgMax (real)](#real).

## Inputs

### $\text{X}$: integer tensor

Input tensor.

#### Constraints

* `[C1]` <a id="C1ix"></a> Axis consistency

  * Statement: The rank $r$ of $X$ shall satisfy `axis < r`.

* `[C2]` <a id="C2ix"></a> Non-empty reduced dimension

  * Statement: The dimension of $X$ along `axis` shall be strictly positive.

## Outputs

### $\text{Y}$: int64 tensor

Output tensor containing the indices of the maximum values of $X$ along `axis`.

#### Constraints

* `[C1]` <a id="C1iy"></a> Output type

  * Statement: $Y$ shall have type `int64`.

* `[C2]` <a id="C2iy"></a> Output shape consistency

  * Statement: See constraint [C2](#C2ry) on tensor $Y$ in [ArgMax (real)](#real).

* `[C3]` <a id="C3iy"></a> Output value range

  * Statement: See constraint [C3](#C3ry) on tensor $Y$ in [ArgMax (real)](#real).

## Numeric accuracy

No numeric accuracy note is required, since **ArgMax** returns integer indices.
