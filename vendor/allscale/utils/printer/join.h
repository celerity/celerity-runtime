#pragma once

namespace allscale {
namespace utils {

	namespace detail {

		template<typename T>
		struct DefaultElementPrinter {
			void operator()(std::ostream& out, const T& value) const {
				out << value;
			}
		};

		template<typename Sep, typename Iter, typename ElementPrinter>
		class joinable {

			Iter begin;
			Iter end;
			Sep sep;
			ElementPrinter printer;

		public:

			joinable(const Iter& begin, const Iter& end, const Sep& sep, const ElementPrinter& printer = ElementPrinter())
				: begin(begin), end(end), sep(sep), printer(printer) {}

			friend
			std::ostream& operator<<(std::ostream& out, const joinable& j) {
				if (j.begin == j.end) return out;
				Iter cur = j.begin;
				j.printer(out, *cur);
				cur++;
				while(cur != j.end) {
					out << j.sep;
					j.printer(out, *cur);
					cur++;
				}
				return out;
			}

		};

	}


	template<typename Iter, typename Value = typename std::iterator_traits<Iter>::value_type>
	detail::joinable<const char*,Iter,detail::DefaultElementPrinter<Value>> join(const char* sep, const Iter& begin, const Iter& end) {
		return detail::joinable<const char*,Iter,detail::DefaultElementPrinter<Value>>(begin,end,sep);
	}

	template<typename Iter, typename Value = typename std::iterator_traits<Iter>::value_type>
	detail::joinable<std::string,Iter,detail::DefaultElementPrinter<Value>> join(const std::string& sep, const Iter& begin, const Iter& end) {
		return detail::joinable<std::string,Iter,detail::DefaultElementPrinter<Value>>(begin,end,sep);
	}

	template<typename Sep, typename Container, typename Iter = typename Container::const_iterator>
	auto join(const Sep& sep, const Container& c) -> decltype(join(sep, c.cbegin(), c.cend())) {
		return join(sep, c.cbegin(), c.cend());
	}

	template<typename Iter, typename Printer>
	detail::joinable<const char*,Iter,Printer> join(const char* sep, const Iter& begin, const Iter& end, const Printer& printer) {
		return detail::joinable<const char*,Iter,Printer>(begin,end,sep,printer);
	}

	template<typename Iter, typename Printer>
	detail::joinable<std::string,Iter,Printer> join(const std::string& sep, const Iter& begin, const Iter& end, const Printer& printer) {
		return detail::joinable<std::string,Iter,Printer>(begin,end,sep,printer);
	}

	template<typename Sep, typename Container, typename Printer, typename Iter = typename Container::const_iterator>
	auto join(const Sep& sep, const Container& c, const Printer& p) -> decltype(join(sep, c.cbegin(), c.cend(),p)) {
		return join(sep, c.cbegin(), c.cend(),p);
	}


} // end namespace utils
} // end namespace allscale
