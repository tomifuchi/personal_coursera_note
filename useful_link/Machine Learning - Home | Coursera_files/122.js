(window.webpackJsonp=window.webpackJsonp||[]).push([[122],{Fg3N:function(module,e,t){"use strict";t.r(e);var n=t("VbXa"),a=t.n(n),r=t("17x9"),o=t.n(r),s=t("q1tI"),i=t.n(s),c=t("YCpq"),u=t("OOvi"),l=t("5E6G"),d=t("L1vm"),h=t("ngyh"),p=t("EVkW"),g=t("bEJl"),m=t("GAn1"),v=t.n(m),f=function(e){function CourseRating(){return e.apply(this,arguments)||this}a()(CourseRating,e);var t=CourseRating.prototype;return t.componentDidMount=function componentDidMount(){this.context.executeAction(u.b,{ratingFeedback:this.props.ratingFeedback})},t.render=function render(){if(!c.a.get("authenticated"))return null;return i.a.createElement("div",{className:"rc-CourseRating"},i.a.createElement(g.a,{course:this.props.course}))},CourseRating}(i.a.Component);f.propTypes={course:o.a.object.isRequired,ratingFeedback:o.a.instanceOf(p.a).isRequired},f.contextTypes={executeAction:o.a.func.isRequired};var C=function(e){function TrackedCourseRating(){return e.apply(this,arguments)||this}a()(TrackedCourseRating,e);var t=TrackedCourseRating.prototype;return t.getChildContext=function getChildContext(){var e=this.props.course,t=e.get?e.get("id"):e.id;return{track:d.a.makeTracker({namespace:"content_learner.rating_course",include:{course_id:t}})}},t.render=function render(){return i.a.createElement(f,this.props)},TrackedCourseRating}(i.a.Component);C.propTypes={course:o.a.object.isRequired,ratingFeedback:o.a.instanceOf(p.a).isRequired},C.childContextTypes={track:o.a.func};var k=function(e){function FluxibleCourseRating(){return e.apply(this,arguments)||this}a()(FluxibleCourseRating,e);var t=FluxibleCourseRating.prototype;return t.componentWillMount=function componentWillMount(){this.fluxibleContext=l.a.createContext()},t.render=function render(){return i.a.createElement(h.a,{context:this.fluxibleContext.getComponentContext()},i.a.createElement(C,this.props))},FluxibleCourseRating}(i.a.Component);e.default=k},GAn1:function(module,exports,e){var t=e("PQP/"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var a={transform:void 0},r=e("aET+")(t,a);t.locals&&(module.exports=t.locals)},"PQP/":function(module,exports,e){},bEJl:function(module,e,t){"use strict";var n=t("pVnL"),a=t.n(n),r=t("VbXa"),o=t.n(r),s=t("q1tI"),i=t.n(s),c=t("17x9"),u=t.n(c),l=t("+eFp"),d=t.n(l),h=t("+OrT"),p=t("OOvi"),g=t("h4VP"),m=t("kvW3"),v=t("8cuT"),f=t("PJ/k"),C=t.n(f),k=t("EVkW"),y=t("esdG"),b=t.n(y),R=t("ltFc"),x=t("ijI/"),S=t("dZ7h"),M=function(e){function CourseRatingContent(){for(var t,n=arguments.length,a=new Array(n),r=0;r<n;r++)a[r]=arguments[r];return(t=e.call.apply(e,[this].concat(a))||this).state={showModal:!1,showSurveyModal:!1,showMessage:!1,message:""},t.handleClick=function(e){var n=t.context.track;e.preventDefault(),n("click.rating"),t.setState({showModal:!0})},t.handleClose=function(){var e;(0,t.context.track)("click.cancel"),t.setState({showModal:!1})},t.handleSubmit=function(e,n){var a=t.context,r=a.track,o=a.executeAction,s=t.props.course,c=t.getCourseNameAndId(s),u=c.courseId,l=c.courseName;t.setState({showModal:!1,showSurveyModal:!0,showMessage:!0,message:i.a.createElement(m.a,{message:b()("Your review of <strong>{courseName}</strong> has been submitted."),courseName:l})}),o(p.a,{courseId:u,value:e,active:!0,comment:n}),r("click.submit",{feedback_length:g.c.getLength(n),feedback_value:e})},t.handleClear=function(){var e=t.context,n=e.track,a=e.executeAction,r=t.props.course,o=t.getCourseNameAndId(r),s=o.courseId,c=o.courseName;t.setState({showModal:!1,showMessage:!0,message:i.a.createElement(m.a,{message:b()("Your review of <strong>{courseName}</strong> has been removed."),courseName:c})}),a(p.a,{courseId:s,value:0,active:!1,comment:g.c.create("")}),n("click.clear")},t.handleMessageTimeout=function(){t.setState({showMessage:!1})},t.handleSurveyContinue=function(){var e=t.context.track,n=t.props,a=n.user,r=n.course,o,s=t.getCourseNameAndId(r).courseId,i="https://www.surveymonkey.com/r/3N5SQ3L?externalUserId=".concat(a.external_id,"&courseId=").concat(s);t.setState({showSurveyModal:!1}),window.open(i,"_blank"),e("survey_modal.click.continue",{user_id:a.external_id})},t.handleSurveyClose=function(){var e=t.context.track,n=t.props.user;t.setState({showSurveyModal:!1}),e("survey_modal.click.close",{user_id:n.external_id})},t}o()(CourseRatingContent,e);var t=CourseRatingContent.prototype;return t.getCourseNameAndId=function getCourseNameAndId(e){var t=!!e.get;return{courseId:t?e.get("id"):e.id,courseName:t?e.get("name"):e.name}},t.render=function render(){var e,t=this.state,n=t.showModal,r=t.showMessage,o=t.message,s=t.showSurveyModal,c=this.props,u=c.course,l=c.ratingFeedback,p=b()("Rate this course"),g=i.a.createElement(f.Tooltip,null,p),m=null==u?void 0:null===(e=u.courseTypeMetadata)||void 0===e?void 0:e.isGuidedProject;if(!l)return i.a.createElement("div",null);return i.a.createElement("div",{className:"rc-CourseRatingContent"},i.a.createElement(f.OverlayTrigger,{placement:"top",overlay:g},i.a.createElement("button",{ref:"icons",type:"button",className:"c-course-rating-icons-container nostyle button-link",onClick:this.handleClick,"aria-label":b()("Rate this course")},i.a.createElement(x.a,{value:l.value}))),n&&i.a.createElement(R.a,a()({},this.props,{onSubmit:this.handleSubmit,onClear:this.handleClear,onClose:this.handleClose})),s&&i.a.createElement(S.a,{onContinue:this.handleSurveyContinue,onClose:this.handleSurveyClose,shouldTakeSurvey:!m}),i.a.createElement(d.a,{transitionName:"fade",transitionEnterTimeout:500,transitionLeaveTimeout:500},r&&i.a.createElement(h.a,{key:"feedback-complete",onTimeout:this.handleMessageTimeout},o)))},CourseRatingContent}(i.a.Component);M.propTypes={course:u.a.object.isRequired,ratingFeedback:u.a.instanceOf(k.a)},M.contextTypes={executeAction:u.a.func.isRequired,track:u.a.func.isRequired},e.a=Object(v.a)(M,["ApplicationStore","CourseRatingStore"],function(e,t){var n=e.ApplicationStore,a;return{ratingFeedback:e.CourseRatingStore.getMyRatingFeedback(),user:n.getUserData()}});var w=M},"ijI/":function(module,e,t){"use strict";var n=t("VbXa"),a=t.n(n),r=t("w/1P"),o=t.n(r),s=t("17x9"),i=t.n(s),c=t("q1tI"),u=t.n(c),l=t("t57v"),d=t.n(l),h=function(e){function CourseRatingIcons(){for(var t,n=arguments.length,a=new Array(n),r=0;r<n;r++)a[r]=arguments[r];return(t=e.call.apply(e,[this].concat(a))||this).state={highlightValue:-1},t.handleMouseLeave=function(){t.setState({highlightValue:-1})},t}a()(CourseRatingIcons,e);var t=CourseRatingIcons.prototype;return t.handleMouseOver=function handleMouseOver(e){this.setState({highlightValue:e})},t.handleSelect=function handleSelect(e){this.setState({highlightValue:-1}),this.props.onSelect&&this.props.onSelect(e)},t.renderIcon=function renderIcon(e,t){var n=this,a=e<=this.state.highlightValue,r=this.state.highlightValue>-1,s=e%1!=0,i=o()("c-course-rating-icon","cif-icon",{"cif-star":a||t&&!r&&!s,"cif-star-half-empty":!a&&t&&!r&&s,"cif-star-o":!a&&(!t||r),highlight:a});return u.a.createElement("i",{key:e,className:i,onMouseOver:!this.props.readOnly&&function(){return n.handleMouseOver(e)},onClick:function onClick(){return n.handleSelect(e)},"aria-hidden":"true"})},t.render=function render(){var e=this,t=o()("rc-CourseRatingIcons",this.props.size,{"read-only":this.props.readOnly}),n=CourseRatingIcons.getRatingDistribution(this.props.value);return u.a.createElement("div",{className:t,onMouseLeave:this.handleMouseLeave,"data-e2e":"course-rating"},n.map(function(t){var n=t.value,a=t.selected;return e.renderIcon(n,a)}))},CourseRatingIcons}(u.a.Component);h.propTypes={onSelect:i.a.func,readOnly:i.a.bool,value:i.a.number.isRequired,size:i.a.oneOf(["small","large"])},h.defaultProps={size:"small",readOnly:!0},h.getRatingDistribution=function(e){var t=.5,n=[],a=Math.round(2*e)/2;for(5!==e&&(a=Math.min(a,4.5));t<=a;){var r;t%1!=0?t+.5>a?(n.push({value:t,selected:!0}),t+=1):t+=.5:(n.push({value:t,selected:!0}),t+=.5)}for(t%1!=0&&(t+=.5);t<=5;)n.push({value:t,selected:!1}),t+=1;return n},e.a=h}}]);
//# sourceMappingURL=122.680ca6967c4e5a854944.js.map