#pragma once

using DataTY = double;

constexpr DataTY operator""_FT(long double value) { return static_cast<DataTY>(value); }