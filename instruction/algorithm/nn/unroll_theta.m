function [unrolled_theta] = unroll_theta(theta)
	%unroll theta cell array into a long-ass vector
	%Not including biased which is theta_i0

	unrolled_theta = [];
	for i=1:length(theta)
		unrolled_theta = [unrolled_theta;theta{i}(:,2:end)(:)];
	endfor
end
