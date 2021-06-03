function unrolled_grad = unroll_grads(grad)
	%unroll grads cell array into a long-ass vector

	unrolled_grad = [];
	for i=1:length(grad)
		unrolled_grad = [unrolled_grad;grad{i}(:)];
	endfor
end
