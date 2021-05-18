<script lang="ts">
	import Menu from '@smui/menu';
	import List, { Item, Separator, Text } from '@smui/list';
	import Button from '@smui/button';
  
	let menu;
	let beer_name="PLACE_HOLDER"
	let img = "./tmp.png"
	let beer_desc = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec gravida tortor eget felis molestie, non luctus quam aliquam. Nunc et arcu est. Nulla at lacus lorem. Mauris faucibus bibendum magna vel facilisis. Integer ut ligula quis sem efficitur luctus. Pellentesque a lacus nec magna mollis dapibus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam in nibh ac magna ultricies mattis. Sed elementum arcu et lacus sollicitudin, a finibus mauris faucibus. Mauris diam arcu, sollicitudin et tortor at, convallis euismod nunc. Donec tempus mauris metus, in tempor orci pharetra eget. Nam euismod massa non egestas lacinia. Ut rutrum quam at mauris interdum, nec dictum ex finibus. Maecenas sit amet congue ipsum. Sed volutpat nisi ut vehicula congue. Donec justo ante, elementum quis ullamcorper sit amet, venenatis vitae tellus."
</script>

	<div style="min-width: 100px;">
		<h1 style="color:white">{beer_name}</h1>
		<img src={img} alt="focused beer">
		<div class="content-scroller">

			<p style="color:white">{beer_desc}</p>
			TUTAJ POWINNIEN BYĆ WYKRES
			
			<img src={img} >
		</div>
		
	  </div>


<style>

	.content-scroller{
		overflow: scroll;
		height: 50%;
	}
</style>